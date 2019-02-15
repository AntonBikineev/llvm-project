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

#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
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
#include <unordered_set>

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

using IntrinsicVector = SmallVector<IntrinsicInst *, 2>;

template <bool IsPostDom>
static decltype(auto) make_directed_range(BasicBlock *BB) {
  if constexpr (IsPostDom)
    return reverse(BB->getInstList());
  else
    return BB->getInstList();
}

// TODO: don't superfluously traverse some instructions in the first BB
template <bool IsPostDom>
static IntrinsicVector findClosestIntrinsicForValue(
    Instruction *FromI, const Value *V, Intrinsic::ID ID, unsigned NumOp,
    const DominatorTreeBase<BasicBlock, IsPostDom> &DT, const DataLayout &DL) {
  IntrinsicVector IntrV;

  for (auto DTN : depth_first(DT.getNode(FromI->getParent()))) {
    if (!DTN) {
      dbgs() << "NO DOMINATOR TREE NODE!\n";
      continue;
    }

    BasicBlock *BB = DTN->getBlock();
    if (!BB) // PostDominatorTree may return empty blocks for exit nodes
      continue;
    // BasicBlock *BB = DTN;

    if (!BB)
      dbgs() << "NO BLOCK!\n";

    if (BB->empty())
      dbgs() << "EMPTY!\n";

    for (Instruction &I : make_directed_range<IsPostDom>(BB)) {
      if (!isa<IntrinsicInst>(I))
        continue;

      auto &II = cast<IntrinsicInst>(I);
      if (II.getIntrinsicID() != ID)
        continue;

      Value *UV = GetUnderlyingObject(II.getOperand(NumOp), DL);
      if (UV == V)
        IntrV.emplace_back(&II);
    }
  }
  return IntrV;
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
  const DominatorTree &DT;
  const PostDominatorTree &PDT;
  LifetimeFrameMap &Lifetimes;

  /*
  template <bool DirDown = true> static auto CreateBBRange(BasicBlock *BB) {
    if constexpr (DirDown)
      return successors(BB);
    else
      return predecessors(BB);
  }

  // TODO: change to dominator tree
  template <bool DirDown = true>
  IntrinsicInst *BFSClosestIntrinsic(Instruction *BI, Value *V,
                                     Intrinsic::ID ID) {
    std::queue<Instruction *> queue;
    std::unordered_set<Instruction *> visited;
    queue.push(BI);

    while (!queue.empty()) {
      Instruction *CI = queue.front();
      queue.pop();

      visited.emplace(CI);

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
  */

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

      IntrinsicVector LFSV = findClosestIntrinsicForValue(
          &Intr, V, Intrinsic::lifetime_start, 1u, PDT, DL);
      if (LFSV.empty())
        return;
      // otherwise, assert that there is a single one
      assert(LFSV.size() == 1 &&
             "How come are there many lifetime.starts for a single value?");

      LFFrame.CtorFrame.Begin = LFSV[0];
    } else if (ID == Intrinsic::cxx_lifetime_end) {
      IntrinsicVector LFEV = findClosestIntrinsicForValue(
          &Intr, V, Intrinsic::lifetime_end, 1u, DT, DL);
      if (LFEV.empty())
        return;
      // otherwise, assert that there is a single one
      assert(LFEV.size() == 1 &&
             "How come are there many lifetime.starts for a single value?");

      auto &Dtors = LFFrame.DtorFrames;
      Dtors.push_back({&Intr, LFEV[0]});
    }
#if 0
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
#endif
  }

public:
  LifetimeTracker(Function &F, const DominatorTree &DT,
                  const PostDominatorTree &PDT, LifetimeFrameMap &Lifetimes)
      : DL{F.getParent()->getDataLayout()}, DT{DT}, PDT{PDT}, Lifetimes{
                                                                  Lifetimes} {
    visit(F);
  }

  // TODO: these checks should be unrecoverable errors - change to asserts
  // The pass may produce some inconsistent results, that's why
  // we currently treat inconsistencies as soft errors.
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
    if (isNonTriviallyAccessedIn(Source->V, Source->CtorFrame.End, CopyIntr))
      return false;

    const CXXFrames &DtorFrames = Source->DtorFrames;
    for (const auto &DF : DtorFrames)
      if (isNonTriviallyAccessedIn(Source->V, Target->CtorFrame.End, DF.Begin))
        return false;

    return true;
  }
};

// maps from copied objects to specific copy data
using CopyMap = SmallDenseMap<Value *, CopyData>;

class CopyTracker : public InstVisitor<CopyTracker> {
  const DataLayout &DL;
  LifetimeFrameMap &Lifetimes;
  CopyMap &Copies;

public:
  CopyTracker(const DataLayout &DL, LifetimeFrameMap &Lifetimes,
              CopyMap &Copies) noexcept
      : DL{DL}, Lifetimes{Lifetimes}, Copies{Copies} {}

  void visitIntrinsicInst(IntrinsicInst &Intr) {
    if (!isCXXCopyInst(Intr))
      return;

    Value *ThisV = GetUnderlyingObject(Intr.getOperand(0), DL);
    Value *FromV = GetUnderlyingObject(Intr.getOperand(1), DL);

    // TODO: remove after investigation
    // assert(Copies.find(ThisV) == Copies.end() &&
    //       "Object can't be copied twice");

    auto ThisIt = Lifetimes.find(ThisV);
    // TODO: remove after investigation
    // assert(ThisIt != Lifetimes.end() && "Lifetime of 'This' is not
    // tracked!");

    auto FromIt = Lifetimes.find(FromV);
    // TODO: remove after investigation
    // assert(FromIt != Lifetimes.end() && "Lifetime of 'From' is not
    // tracked!");

    // TODO: remove after investigation
    if (ThisIt != Lifetimes.end() && FromIt != Lifetimes.end())
      Copies.insert({ThisV, {&Intr, &ThisIt->second, &FromIt->second}});
  }
};

void statLifetimes(const Function &F, const LifetimeFrameMap &LFMap) {
  LLVM_DEBUG(dbgs() << "Function " << F.getName() << " has " << LFMap.size()
                    << " cxx.lifetimes\n");
  for (const auto &P : LFMap) {
    const auto &DtorFrames = P.second.DtorFrames;
    LLVM_DEBUG(dbgs() << "\tValue '" << *P.first << "' has "
                      << DtorFrames.size() << " dtor frames\n");
  }
}

void statCopies(const Function &F, const CopyMap &CMap) {
  LLVM_DEBUG(dbgs() << "Function " << F.getName() << " has " << CMap.size()
                    << " cxx copies\n");
  for (const auto &P : CMap) {
    const Value *Target = P.second.Target->V;
    const Value *Source = P.second.Source->V;
    LLVM_DEBUG(dbgs() << "\tValue '" << *Target << "' copies value '" << *Source
                      << "'\n");
  }
}

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
  // we may leave the split BB as is, but it seems to be better to
  // explicitly remove it to ensure that it has no uses
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
  Target->replaceAllUsesWith(Source);
}

bool tryEliminatingSingleCopy(Function &F, IntrinsicInst &Intr,
                              DominatorTree &DT, PostDominatorTree &PDT) {
  assert(isCXXCopyInst(Intr) &&
         "Not a copy instruction; maybe iteration invalidation took place");

  const DataLayout &DL = F.getParent()->getDataLayout();

  Value *Target = GetUnderlyingObject(Intr.getOperand(0), DL);
  Value *Source = GetUnderlyingObject(Intr.getOperand(1), DL);

  LifetimeFrameMap Lifetimes;
  LifetimeTracker LFT{F, DT, PDT, Lifetimes};
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

  elideCXXCopy(CD);

#if 0
  // find lifetime for the Target Value
  IntrinsicVector LFSV = findClosestIntrinsicForValue(
      &Intr, Target, Intrinsic::lifetime_start, 1u, PDT, DL);
  if (LFSV.empty()) {
    LLVM_DEBUG(
        dbgs() << "Error: we haven't found any lifetime.start for '"
               << *Target << "', ignore the copy\n");
    return false;
  }
  // otherwise, assert that there is a single one
  assert(LFSV.size() == 1 && "How come are there many lifetime.starts for a single value?");

  IntrinsicVector LFEV = findClosestIntrinsicForValue(
      &Intr, Target, Intrinsic::lifetime_start, 1u, PDT, DL);
  if (LFEV.empty()) {
    LLVM_DEBUG(
        dbgs() << "Error: we haven't found any cxx.lifetime.start for '"
               << *Target << "', ignore the copy\n");
    return false;
  }
  // otherwise, assert that there is a single one
  assert(LFSV.size() == 1 && "How come are there many cxx.lifetime.starts for a single value?");
#endif

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

IntrinsicInst *findUnvisitedCXXCopyIntrinsic(const DominatorTree &DT,
                                             const DataLayout &DL,
                                             VisitedCopies &VCS) {
  for (auto DTN : depth_first(DT.getRootNode())) {
    for (Instruction &I : *DTN->getBlock())
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

    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto &PDT = getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();

    const DataLayout &DL = F.getParent()->getDataLayout();
    VisitedCopies VCS;

    // we have to deeply eliminate copies; TODO: elaborate more on this
    while (auto *CopyIntr = findUnvisitedCXXCopyIntrinsic(DT, DL, VCS)) {
      CopyIdentity CI{*CopyIntr, DL};
      VCS.insert({*CopyIntr, DL});
      if (!tryEliminatingSingleCopy(F, *CopyIntr, DT, PDT))
        continue;
      DT.recalculate(F);
      PDT.recalculate(F);
      changed = true;
      // update statistics
      ++CXXCopyElimCounter;
    }

    return changed;
  }

#if 0
    LifetimeFrameMap Lifetimes;
    LifetimeTracker LFT(F.getParent()->getDataLayout(), Lifetimes);

    LFT.visit(F);

    if (!verifyLifetimes(Lifetimes))
      return false;

    statLifetimes(F, Lifetimes);

    CopyMap Copies;
    CopyTracker CT(F.getParent()->getDataLayout(), Lifetimes, Copies);

    CT.visit(F);

    statCopies(F, Copies);

    // TODO: DEBUG
    if (!Copies.empty())
      LLVM_DEBUG(dbgs() << "DUMPING FUNCTION BEFORE\n"; F.dump());

    const bool changed = elideCXXCopies(Copies);
    // TODO: DEBUG
    if (changed)
      LLVM_DEBUG(dbgs() << "DUMPING FUNCTION AFTER\n"; F.dump());

    return changed;
  }
#endif

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<PostDominatorTreeWrapperPass>();
    // TODO: the analysis is quite aggressive, so we
    // probably shouldn't preserve anything
    // TODO: don't recalculate domtree each time,
    // use DomUpdater
  }
};
} // namespace

FunctionPass *llvm::createCXXCopyEliminationPass() {
  PassRegistry &Registry = *PassRegistry::getPassRegistry();
  initializeDominatorTreeWrapperPassPass(Registry);
  initializePostDominatorTreeWrapperPassPass(Registry);
  return new CXXCopyElimPass();
}

char CXXCopyElimPass::ID = 0;
static RegisterPass<CXXCopyElimPass> X("cxxcopyelim", "Eliminate C++ copies");

static void registerMyPass(const llvm::PassManagerBuilder &,
                           legacy::PassManagerBase &PM) {
  PassRegistry &Registry = *PassRegistry::getPassRegistry();
  initializeDominatorTreeWrapperPassPass(Registry);
  initializePostDominatorTreeWrapperPassPass(Registry);
  PM.add(new CXXCopyElimPass());
}

static llvm::RegisterStandardPasses
    RegisterMyPassEarly(llvm::PassManagerBuilder::EP_EarlyAsPossible,
                        registerMyPass);
static llvm::RegisterStandardPasses
    RegisterMyPassAfter(llvm::PassManagerBuilder::EP_ScalarOptimizerLate,
                        registerMyPass);
