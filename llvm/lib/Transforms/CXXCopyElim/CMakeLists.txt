# If we don't need RTTI or EH, there's no reason to export anything
# from the CXXCopyElim plugin.
if( NOT LLVM_REQUIRES_RTTI )
  if( NOT LLVM_REQUIRES_EH )
    set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/CXXCopyElim.exports)
  endif()
endif()

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS Core Support)
endif()

add_llvm_library( LLVMCXXCopyElim MODULE
  CXXCopyElim.cpp

  DEPENDS
  intrinsics_gen
  PLUGIN_TOOL
  opt
  )

add_llvm_library( LLVMCXXCopyElimStatic
  CXXCopyElim.cpp

  DEPENDS
  intrinsics_gen
  )
