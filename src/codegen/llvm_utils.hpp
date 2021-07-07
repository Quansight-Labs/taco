#pragma once

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include "taco/error.h"

inline llvm::Type* get_int_type(const int width, llvm::LLVMContext& Context) {
  switch (width) {
    case 64:
      return llvm::Type::getInt64Ty(Context);
    case 32:
      return llvm::Type::getInt32Ty(Context);
    case 16:
      return llvm::Type::getInt16Ty(Context);
    case 8:
      return llvm::Type::getInt8Ty(Context);
    case 1:
      return llvm::Type::getInt1Ty(Context);
  }
  taco_ierror << "Unsupported width value " << width;
  return nullptr;
}

inline llvm::Type* get_fp_type(const int width, llvm::LLVMContext& Context) {
  switch (width) {
    case 64:
      return llvm::Type::getDoubleTy(Context);
    case 32:
      return llvm::Type::getFloatTy(Context);
  }
  taco_ierror << "Unsupported width value " << width;
  return nullptr;
}

inline llvm::Type* get_fp_ptr_type(const int width, llvm::LLVMContext& Context) {
  switch (width) {
    case 64:
      return llvm::Type::getDoublePtrTy(Context);
    case 32:
      return llvm::Type::getFloatPtrTy(Context);
  }
  taco_ierror << "Unsupported width value " << width;
  return nullptr;
}

inline llvm::Type* get_int_ptr_type(const int width, llvm::LLVMContext& Context) {
  switch (width) {
    case 64:
      return llvm::Type::getInt64PtrTy(Context);
    case 32:
      return llvm::Type::getInt32PtrTy(Context);
    case 16:
      return llvm::Type::getInt16PtrTy(Context);
    case 8:
      return llvm::Type::getInt8PtrTy(Context);
    case 1:
      return llvm::Type::getInt1PtrTy(Context);
  }
  taco_ierror << "Unsupported width value " << width;
  return nullptr;
}

inline llvm::Type* get_void_type(llvm::LLVMContext& Context) {
  return llvm::Type::getVoidTy(Context);
}

inline llvm::Type* get_void_ptr_type(llvm::LLVMContext& Context) {
  return get_int_ptr_type(8, Context);
}

inline llvm::Value* get_int_constant(const int width,
                                     const int value,
                                     llvm::LLVMContext& Context) {
  auto typ = get_int_type(width, Context);
  return llvm::ConstantInt::get(typ, value);
}
