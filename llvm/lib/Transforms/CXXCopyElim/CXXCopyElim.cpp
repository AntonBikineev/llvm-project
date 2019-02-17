//===- CXXCopyElim.cpp - Eliminate CXX copies ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// TBD
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils.h"

#include <queue>

using namespace llvm;

#define DEBUG_TYPE "cxxcopyelim"

STATISTIC(CXXCopyElimCounter, "Number of eliminated C++ copies");

namespace {

bool isIdleInstruction(const Instruction *I) {
  if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
    if (!II->isLifetimeStartOrEnd() && !II->isCXXLifetimeOrCopy())
      return false;
  } else if (const BitCastInst *BCI = dyn_cast<BitCastInst>(I)) {
    if (BCI->getType() != Type::getInt8PtrTy(I->getContext()))
      return false;
    if (!onlyUsedByLifetimeOrCXXMarkers(BCI))
      return false;
  } else if (const GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(I)) {
    if (GEPI->getType() != Type::getInt8PtrTy(I->getContext()))
      return false;
    if (!GEPI->hasAllZeroIndices())
      return false;
    if (!onlyUsedByLifetimeOrCXXMarkers(GEPI))
      return false;
  } else {
    return false;
  }

  return true;
}

bool isCXXLifetimeInst(const Instruction &I) {
  const IntrinsicInst &II = cast<const IntrinsicInst>(I);
  return II.getIntrinsicID() == Intrinsic::cxx_lifetime_start ||
         II.getIntrinsicID() == Intrinsic::cxx_lifetime_end;
}

bool isCXXCopyInst(const Instruction &I) {
  const IntrinsicInst &II = cast<const IntrinsicInst>(I);
  return II.getIntrinsicID() == Intrinsic::cxx_copy;
}

bool isInBetweenOf(const Instruction *I, const Instruction *From,
                   const Instruction *To) {
  // TODO: make use of dominator tree later on to speed things up
  if (!isPotentiallyReachable(From, I))
    return false;

  if (!isPotentiallyReachable(I, To))
    return false;

  return true;
}

bool isNonTriviallyAccessedIn(const Value *V, const Instruction *From,
                              const Instruction *To) {
  for (const auto U : V->users()) {
    // assumes yes, if it's constant or memory access or whatever
    if (!isa<Instruction>(U))
      return true;
    auto *I = cast<const Instruction>(U);
    if (isInBetweenOf(I, From, To) && !isIdleInstruction(I))
      return true;
  }

  return false;
}

struct CXXFrame {
  IntrinsicInst *Begin = nullptr; // lifetime.start
  IntrinsicInst *End = nullptr;   // cxx.lifetime.start
};

using CXXFrames = SmallVector<CXXFrame, 2>;

// structure that keeps track of CXX lifetime for a value V
struct LifetimeFrame {
  Value *V;
  CXXFrame CtorFrame;
  CXXFrames DtorFrames; // there may be many dtors in CFG
                        // generated for a single object
                        // (e.g. for normal and EH BBs)

  explicit LifetimeFrame(Value *V) noexcept : V(V) {}
};

using LifetimeFrameMap = SmallDenseMap<Value *, LifetimeFrame>;

class LifetimeTracker : private InstVisitor<LifetimeTracker> {
  friend class InstVisitor<LifetimeTracker>;

  const DataLayout &DL;
  LifetimeFrameMap &Lifetimes;

  template <bool DirDown = true> static auto CreateBBRange(BasicBlock *BB) {
    if constexpr (DirDown)
      return successors(BB);
    else
      return predecessors(BB);
  }

  template <bool DirDown = true>
  IntrinsicInst *BFSClosestIntrinsic(Instruction *BI, Value *V,
                                     Intrinsic::ID ID) {
    std::queue<Instruction *> queue;
    SmallPtrSet<Instruction *, 8> visited;
    queue.push(BI);

    while (!queue.empty()) {
      Instruction *CI = queue.front();
      queue.pop();

      visited.insert(CI);

      // base case
      if (auto *II = dyn_cast<IntrinsicInst>(CI)) {
        if (II->getIntrinsicID() == ID) {
          if (GetUnderlyingObject(II->getOperand(1), DL) == V)
            return II;
        }
      }

      if constexpr (DirDown) {
        if (!CI->isTerminator()) {
          if (visited.find(CI->getNextNode()) == visited.end())
            queue.push(CI->getNextNode());
          continue;
        }
      } else {
        if (CI->getPrevNode()) {
          if (visited.find(CI->getPrevNode()) == visited.end())
            queue.push(CI->getPrevNode());
          continue;
        }
      }

      // otherwise traverse other basic blocks
      auto BBRange = CreateBBRange<DirDown>(CI->getParent());
      for (BasicBlock *BB : BBRange)
        if (!BB->empty())
          queue.push(DirDown ? &BB->front() : &BB->back());
    }

    return nullptr;
  }

  void visitIntrinsicInst(IntrinsicInst &Intr) {
    if (!isCXXLifetimeInst(Intr))
      return;

    Value *V = GetUnderlyingObject(Intr.getOperand(1), DL);

    auto It = Lifetimes.find(V);
    if (It == Lifetimes.end())
      It = Lifetimes.insert({V, LifetimeFrame{V}}).first;

    auto &LFFrame = It->second;

    if (const Intrinsic::ID ID = Intr.getIntrinsicID();
        ID == Intrinsic::cxx_lifetime_start) {
      LFFrame.CtorFrame.End = &Intr;

      IntrinsicInst *Begin =
          BFSClosestIntrinsic<false>(&Intr, V, Intrinsic::lifetime_start);
      // TODO: remove after investigation
      // assert(Begin && "Couldn't find corresponding lifetime.start
      // intrinsic");

      LFFrame.CtorFrame.Begin = Begin;
    } else if (ID == Intrinsic::cxx_lifetime_end) {
      IntrinsicInst *End =
          BFSClosestIntrinsic<true>(&Intr, V, Intrinsic::lifetime_end);
      // TODO: remove after investigation
      // assert(End && "Couldn't find corresponding lifetime.end intrinsic");

      auto &Dtors = LFFrame.DtorFrames;
      Dtors.push_back({&Intr, End});
    }
  }

public:
  LifetimeTracker(Function &F, LifetimeFrameMap &Lifetimes)
      : DL{F.getParent()->getDataLayout()}, Lifetimes{Lifetimes} {
    visit(F);
  }

  // TODO: these checks should eventually be unrecoverable errors -
  // change to asserts. The pass may produce some inconsistent results,
  // that's why we currently treat inconsistencies as soft errors.
  bool verify() const noexcept {
    bool failed = false;
    for (const auto &P : Lifetimes) {
      const Value *V = P.first;
      const auto &Ctor = P.second.CtorFrame;
      if (!Ctor.Begin) {
        LLVM_DEBUG(
            dbgs() << "Error: no corresponding lifetime.start intrinsic for '"
                   << *V << "'\n");
        failed = true;
      }
      if (!Ctor.End) {
        LLVM_DEBUG(
            dbgs()
            << "Error: no corresponding cxx.lifetime.start intrinsic for '"
            << *V << "'\n");
        failed = true;
      }

      const auto &Dtors = P.second.DtorFrames;
      if (Dtors.empty()) {
        LLVM_DEBUG(dbgs() << "Error: no corresponding dtor frame for '" << *V
                          << "'\n");
        failed = true;
      }

      for (const auto &Dtor : P.second.DtorFrames) {
        if (!Dtor.Begin) {
          LLVM_DEBUG(
              dbgs()
              << "Error: no corresponding cxx.lifetime.end intrinsic for '"
              << *V << "'\n");
          failed = true;
        }
        if (!Dtor.End) {
          LLVM_DEBUG(dbgs()
                     << "Error: no corresponding lifetime.end intrinsic for '"
                     << *V << "'\n");
          failed = true;
        }
      }
    }
    return !failed;
  }
};

struct CopyData {
  IntrinsicInst *CopyIntr = nullptr; // cxx.copy
  LifetimeFrame *Target = nullptr;   // 'This' object
  LifetimeFrame *Source = nullptr;   // 'From' object

  bool canBeElided() const {
    // TODO: we don't currently support the following case,
    // but have to address it later on
    if (Target->V->getType() != Source->V->getType())
      return false;

    if (isNonTriviallyAccessedIn(Source->V, Source->CtorFrame.End, CopyIntr))
      return false;

    const CXXFrames &DtorFrames = Source->DtorFrames;
    for (const auto &DF : DtorFrames)
      if (isNonTriviallyAccessedIn(Source->V, Target->CtorFrame.End, DF.Begin))
        return false;

    return true;
  }
};

bool hasUsesAfter(const Value *V, const Instruction *AfterI) {
  for (const User *U : V->users())
    if (isa<Instruction>(U) &&
        isPotentiallyReachable(AfterI, cast<Instruction>(U)))
      return true;

  return false;
}

SmallVector<Value *, 8> getUsedDefsToBeMoved(Instruction *BeginI,
                                             Instruction *EndI) {
  SmallVector<Value *, 8> Results;
  SmallVector<BasicBlock *, 4> Worklist;

  Instruction *I = BeginI;
  for (; I != EndI && !I->isTerminator(); I = I->getNextNode())
    if (hasUsesAfter(I, EndI))
      Results.push_back(I);

  if (I == EndI)
    return Results;

  // I is terminator
  BasicBlock *BB = I->getParent();
  Worklist.append(succ_begin(BB), succ_end(BB));

  while (!Worklist.empty()) {
    BB = Worklist.pop_back_val();
    I = &BB->front();
    for (; I != EndI && !I->isTerminator(); I = I->getNextNode())
      if (hasUsesAfter(I, EndI))
        Results.push_back(I);
    if (I->isTerminator())
      Worklist.append(succ_begin(BB), succ_end(BB));
  }

  return Results;
}

// removes code by splitting basic blocks for Begin and End instruction;
// function is assumed to work for Begin and End belonging to the same
// BB as well as to different BBs; function accepts semi-open interval
// TODO: we should address EndI == nullptr
void removeCodeBySplittingBlocks(Instruction *BeginI, Instruction *EndI) {
  assert(BeginI && "BeginI is nullptr");
  assert(EndI && "EndI is nullptr");

  auto Defs = getUsedDefsToBeMoved(BeginI, EndI);
  // dbgs() << "****************SIZEOF USEDTOBEMOVED: " << Defs.size() << "\n";
  // for (const auto& U: Defs)
  // dbgs() << "\tDef: " << *U << "\n";

  BasicBlock *EndBB = EndI->getParent();
  BasicBlock *NewEndBB = EndBB->splitBasicBlock(EndI);

  BasicBlock *BeginBB = BeginI->getParent();
  BasicBlock *ToBeRemoved = BeginBB->splitBasicBlock(BeginI);

  // TODO: maybe it would make sense to call changeToUnreachable
  // in the dead BasicBlock

  // check if there are any defs that are used later in the CFG
  // (e.g. phi-instructions from landing pads - the ones we've encountered);
  // if there are, insert them to the front of the NewEndBB
  /*SmallVector<Value *, 8> WorkList;

  for (auto &I : *ToBeRemoved) {
    if (!I.isTerminator() && !I.use_empty()) {
      // dbgs() << "**************************NOT EMPTY\n";
      I.moveBefore(&NewEndBB->front());
    }
  }
  for (auto &I : *EndBB) {
    if (!I.isTerminator() && !I.use_empty()) {
      dbgs() << "***Inserting " << I << "\n";
      I.moveBefore(&NewEndBB->front());
    }
  }
  // dbgs() << "****************SIZEOF ToBeRemoved after: " <<
  // ToBeRemoved->size() << '\n';
  */

  BeginBB->getTerminator()->eraseFromParent();
  BranchInst::Create(NewEndBB, BeginBB);

  /*
  // we may leave the work of removing the split block to DCE,
  // but it seems to be better to explicitly remove it to
  // ensure that it has no uses
  ToBeRemoved->eraseFromParent();
  */
}

void elideCXXCopy(CopyData &CD) {
  Value *Source = CD.Source->V;

  CXXFrame &CF = CD.Target->CtorFrame;
  CXXFrames &DFs = CD.Target->DtorFrames;

  // TODO: check that next node is not nullptr
  assert(CF.End->getNextNode() && "no next node for CF");

  // TODO: simplify
  // TODO: maybe we should remove code starting at copy intrinsic?
  removeCodeBySplittingBlocks(CF.Begin, CF.End->getNextNode());
  for (auto &DF : DFs) {
    assert(DF.End->getNextNode() && "no next node 2");
    removeCodeBySplittingBlocks(DF.Begin, DF.End->getNextNode());
  }

  // replace all the remaining uses of Target with Source
  Value *Target = CD.Target->V;
  LLVM_DEBUG(dbgs() << "Eliding copy with target type: " << *Target->getType()
                    << '\n');
  LLVM_DEBUG(dbgs() << "                  source type: " << *Source->getType()
                    << '\n');

  // check if target and source types are different, and
  // in case they are, cast the latter to the type of the former
  // TODO: this case is currently disabled in canBeElided function
  if (Target->getType() != Source->getType())
    Source = CastInst::CreateBitOrPointerCast(
        Source, Target->getType(), "",
        cast<Instruction>(Source)->getNextNode());

  Target->replaceAllUsesWith(Source);
}

bool tryEliminatingSingleCopy(Function &F, IntrinsicInst &Intr) {
  assert(isCXXCopyInst(Intr) &&
         "Not a copy instruction; maybe iteration invalidation took place");

  const DataLayout &DL = F.getParent()->getDataLayout();

  Value *Target = GetUnderlyingObject(Intr.getOperand(0), DL);
  Value *Source = GetUnderlyingObject(Intr.getOperand(1), DL);

  LifetimeFrameMap Lifetimes;
  LifetimeTracker LFT{F, Lifetimes};
  if (!LFT.verify())
    return false;

  auto ThisIt = Lifetimes.find(Target);
  if (ThisIt == Lifetimes.end()) {
    LLVM_DEBUG(dbgs() << "Error: couldn't find lifetime for '" << *Target
                      << "', ignore the copy\n");
    return false;
  }

  auto FromIt = Lifetimes.find(Source);
  if (FromIt == Lifetimes.end()) {
    LLVM_DEBUG(dbgs() << "Error: couldn't find lifetime for '" << *Source
                      << "', ignore the copy\n");
    return false;
  }

  CopyData CD{&Intr, &ThisIt->second, &FromIt->second};
  if (!CD.canBeElided()) {
    LLVM_DEBUG(dbgs() << "The value '" << *Source
                      << "' is accessed, no way to elide it\n");
    return false;
  }

  LLVM_DEBUG(
      dbgs() << "The value '" << *Source
             << "' is *not* accessed, we are proceeding with the elision\n");

  //LLVM_DEBUG(dbgs() << "**********BEFORE COPY ELIM***********\n" << F);
  elideCXXCopy(CD);
  //LLVM_DEBUG(dbgs() << "***********AFTER COPY ELIM***********\n" << F);

  return true;
}

// We identify copies by their sources and targets.
// It might be not strictly correct, but at this point
// I don't think we invalidate any blocks with allocas
class CopyIdentity {
  Value *Target = nullptr;
  Value *Source = nullptr;

public:
  CopyIdentity(const IntrinsicInst &II, const DataLayout &DL) {
    assert(isCXXCopyInst(II) &&
           "Not a copy instruction; maybe iteration invalidation took place");
    Target = GetUnderlyingObject(II.getOperand(0), DL);
    Source = GetUnderlyingObject(II.getOperand(1), DL);
  }

  const Value *getTarget() const { return Target; }
  const Value *getSource() const { return Source; }

  friend bool operator<(const CopyIdentity &left, const CopyIdentity &right) {
    return std::tie(left.Target, left.Source) <
           std::tie(right.Target, right.Source);
  }
  friend bool operator==(const CopyIdentity &left, const CopyIdentity &right) {
    return std::tie(left.Target, left.Source) <
           std::tie(right.Target, right.Source);
  }
};

using VisitedCopies = SmallSet<CopyIdentity, 4>;

IntrinsicInst *findUnvisitedCXXCopyIntrinsic(Function &F, const DataLayout &DL,
                                             VisitedCopies &VCS) {
  for (BasicBlock *BB : depth_first(&F)) {
    for (Instruction &I : *BB)
      if (isa<IntrinsicInst>(I) && isCXXCopyInst(I)) {
        auto &II = cast<IntrinsicInst>(I);
        if (VCS.count({II, DL}))
          continue;
        return &cast<IntrinsicInst>(I);
      }
  }
  return nullptr;
}

// the pass itself
struct CXXCopyElimPass : public FunctionPass {
  static char ID;
  CXXCopyElimPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;

    bool changed = false;

    const DataLayout &DL = F.getParent()->getDataLayout();
    VisitedCopies VCS;

    // we have to deeply eliminate copies; TODO: elaborate more on this
    while (auto *CopyIntr = findUnvisitedCXXCopyIntrinsic(F, DL, VCS)) {
      CopyIdentity CI{*CopyIntr, DL};
      VCS.insert({*CopyIntr, DL});
      if (!tryEliminatingSingleCopy(F, *CopyIntr))
        continue;
      changed = true;
      // update statistics
      ++CXXCopyElimCounter;
    }

    return changed;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // the analysis is quite aggressive, so we
    // probably shouldn't preserve anything
  }
};
} // namespace

FunctionPass *llvm::createCXXCopyEliminationPass() {
  return new CXXCopyElimPass();
}

char CXXCopyElimPass::ID = 0;
static RegisterPass<CXXCopyElimPass> X("cxxcopyelim", "Eliminate C++ copies");

static void registerMyPass(const llvm::PassManagerBuilder &,
                           legacy::PassManagerBase &PM) {
  PM.add(new CXXCopyElimPass());
}

static llvm::RegisterStandardPasses RegisterMyPassEarly(
    llvm::PassManagerBuilder::EP_EarlyAsPossibleAfterInliner, registerMyPass);
