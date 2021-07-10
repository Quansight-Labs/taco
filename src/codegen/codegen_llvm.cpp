#ifdef HAVE_LLVM
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Constants.h>

#include "codegen_llvm.h"
#include "llvm_utils.hpp"
#include "taco/util/print.h"

using namespace std;

namespace taco {
namespace ir {

class CodeGen_LLVM::FindVars : public IRVisitor {
 public:
  map<Expr, string, ExprCompare> varMap;

  // the variables for which we need to add declarations
  map<Expr, string, ExprCompare> varDecls;

  vector<Expr> localVars;

  // this maps from tensor, property, mode, index to the unique var
  map<tuple<Expr, TensorProperty, int, int>, string> canonicalPropertyVar;

  // this is for convenience, recording just the properties unpacked
  // from the output tensor so we can re-save them at the end
  map<tuple<Expr, TensorProperty, int, int>, string> outputProperties;

  // TODO: should replace this with an unordered set
  vector<Expr> inputTensors;
  vector<Expr> outputTensors;

  bool inBlock;

  CodeGen_LLVM* codeGen;

  // copy inputs and outputs into the map
  FindVars(vector<Expr> inputs, vector<Expr> outputs, CodeGen_LLVM* codeGen) : codeGen(codeGen) {
    for (auto v : inputs) {
      auto var = v.as<Var>();
      taco_iassert(var) << "Inputs must be vars in codegen";
      taco_iassert(varMap.count(var) == 0) << "Duplicate input found in codegen: " << var->name;
      inputTensors.push_back(v);
      varMap[var] = var->name;
    }
    for (auto v : outputs) {
      auto var = v.as<Var>();
      taco_iassert(var) << "Outputs must be vars in codegen";
      taco_iassert(varMap.count(var) == 0) << "Duplicate output found in codegen";

      outputTensors.push_back(v);
      varMap[var] = var->name;
    }
    inBlock = false;
  }

 protected:
  using IRVisitor::visit;

  virtual void visit(const For* op) {
    llvm::errs() << "LLVM FindVars Visiting For\n";
    if (!util::contains(localVars, op->var)) {
      localVars.push_back(op->var);
    }
    op->var.accept(this);
    op->start.accept(this);
    op->end.accept(this);
    op->increment.accept(this);
    op->contents.accept(this);
  }

  virtual void visit(const Var* op) {
    llvm::errs() << "LLVM FindVars Visiting Var \"" << op->name << "\"\n";
    if (varMap.count(op) == 0 && !inBlock) {
      varMap[op] = codeGen->genUniqueName(op->name);
    }
  }

  virtual void visit(const VarDecl* op) {
    llvm::errs() << "LLVM FindVars Visiting VarDecl\n";
    if (!util::contains(localVars, op->var) && !inBlock) {
      localVars.push_back(op->var);
    }
    op->var.accept(this);
    op->rhs.accept(this);
  }

  virtual void visit(const GetProperty* op) {
    if (varMap.count(op) == 0 && !inBlock) {
      auto key = tuple<Expr, TensorProperty, int, int>(
          op->tensor, op->property, (size_t) op->mode, (size_t) op->index);
      if (canonicalPropertyVar.count(key) > 0) {
        varMap[op] = canonicalPropertyVar[key];
      } else {
        auto unique_name = codeGen->genUniqueName(op->name);
        canonicalPropertyVar[key] = unique_name;
        varMap[op] = unique_name;
        varDecls[op] = unique_name;
        if (util::contains(outputTensors, op->tensor)) {
          outputProperties[key] = unique_name;
        }
      }
    }
  }
};

void CodeGen_LLVM::pushSymbol(const std::string& name, llvm::Value* v) {
  this->symbolTable.insert({name, v});
}

void CodeGen_LLVM::removeSymbol(const std::string& name) {
  this->symbolTable.remove(name);
}

llvm::Value* CodeGen_LLVM::getSymbol(const std::string& name) {
  return this->symbolTable.get(name);
}

void CodeGen_LLVM::pushScope() {
  this->symbolTable.scope();
}

void CodeGen_LLVM::popScope() {
  this->symbolTable.unscope();
}

// Convert from taco type to LLVM type
llvm::Type* CodeGen_LLVM::llvmTypeOf(Datatype t) {
  taco_tassert(!t.isComplex()) << "LLVM codegen for complex not yet supported";

  if (t.isFloat()) {
    switch (t.getNumBits()) {
      case 32:
        return llvm::Type::getFloatTy(*this->Context);
      case 64:
        return llvm::Type::getDoubleTy(*this->Context);
      default:
        taco_ierror << "Unable to find LLVM type for " << t;
        return nullptr;
    }
  } else if (t.isInt()) {
    return llvm::Type::getIntNTy(*this->Context, t.getNumBits());
  } else {
    taco_ierror << "Unable to find llvm type for " << t;
    return nullptr;
  }
}

void CodeGen_LLVM::writeModuleToFile(std::string& fileName) {
  std::error_code EC;
  llvm::raw_fd_ostream outputStream(fileName, EC);
  llvm::WriteBitcodeToFile(*this->Module, outputStream);
  outputStream.flush();
}

void CodeGen_LLVM::dumpModule() const {
  taco_iassert(this->Module != nullptr);
  llvm::outs() << *this->Module << "\n";
}

void CodeGen_LLVM::emitPrintf(const std::string& fmt, const std::vector<llvm::Value*>& args) {
  auto* ptr = this->Builder->CreateGlobalStringPtr(fmt);

  auto* i8p = get_int_ptr_type(8, *this->Context);
  auto* i32 = get_int_type(32, *this->Context);

  std::vector<llvm::Type*> argTypes = {i8p};
  std::vector<llvm::Value*> args_ = {ptr};
  for (auto* arg : args) {
    argTypes.emplace_back(arg->getType());
    args_.emplace_back(arg);
  }
  emitExternalCall("printf", i32, argTypes, args_);
  emitfflush();
}

void CodeGen_LLVM::emitfflush() {
  auto* i8p = get_int_ptr_type(8, *this->Context);
  auto* i32 = get_int_type(32, *this->Context);
  auto* null = llvm::ConstantPointerNull::get(i8p);
  emitExternalCall("fflush", i32, {i8p}, {null});
}

llvm::Value* CodeGen_LLVM::emitExternalCall(const std::string& funcName,
                                            llvm::Type* returnType,
                                            const std::vector<llvm::Type*>& argTypes,
                                            const std::vector<llvm::Value*>& args) {
  // Build Function type
  auto FnTy = llvm::FunctionType::get(returnType, argTypes, false);

  // Declare it in the module
  auto func = this->Module->getOrInsertFunction(funcName, FnTy);

  // Call it
  return this->Builder->CreateCall(func, args);
}

void CodeGen_LLVM::compile(Stmt stmt, bool isFirst) {
  init_codegen();
  stmt.accept(this);
}

void CodeGen_LLVM::codegen(Stmt stmt) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "stmt");
  value = nullptr;
  stmt.accept(this);
}

llvm::Value* CodeGen_LLVM::codegen(Expr expr) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Expr");
  value = nullptr;
  expr.accept(this);
  taco_iassert(value) << "Codegen of expression " << expr << " did not produce an LLVM value";
  return value;
}

void CodeGen_LLVM::visit(const Literal* e) {
  if (e->type.isFloat()) {
    if (e->type.getNumBits() == 32) {
      value = llvm::ConstantFP::get(llvmTypeOf(e->type), e->getValue<float>());
    } else {
      value = llvm::ConstantFP::get(llvmTypeOf(e->type), e->getValue<double>());
    }
  } else if (e->type.isUInt()) {
    switch (e->type.getNumBits()) {
      case 8:
        value = llvm::ConstantInt::get(llvmTypeOf(e->type), e->getValue<uint8_t>());
        return;
      case 16:
        value = llvm::ConstantInt::get(llvmTypeOf(e->type), e->getValue<uint16_t>());
        return;
      case 32:
        value = llvm::ConstantInt::get(llvmTypeOf(e->type), e->getValue<uint32_t>());
        return;
      case 64:
        value = llvm::ConstantInt::get(llvmTypeOf(e->type), e->getValue<uint64_t>());
        return;
      case 128:
        value = llvm::ConstantInt::get(llvmTypeOf(e->type), e->getValue<unsigned long long>());
        return;
      default:
        taco_ierror << "Unable to generate LLVM for literal " << e;
    }
  } else if (e->type.isInt()) {
    switch (e->type.getNumBits()) {
      case 8:
        value = llvm::ConstantInt::getSigned(llvmTypeOf(e->type), e->getValue<int8_t>());
        return;
      case 16:
        value = llvm::ConstantInt::getSigned(llvmTypeOf(e->type), e->getValue<int16_t>());
        return;
      case 32:
        value = llvm::ConstantInt::getSigned(llvmTypeOf(e->type), e->getValue<int32_t>());
        return;
      case 64:
        value = llvm::ConstantInt::getSigned(llvmTypeOf(e->type), e->getValue<int64_t>());
        return;
      case 128:
        value = llvm::ConstantInt::getSigned(llvmTypeOf(e->type), e->getValue<long long>());
        return;
      default:
        taco_ierror << "Unable to generate LLVM for literal " << e;
    }
  } else {
    taco_ierror << "Unable to generate LLVM for literal " << e;
  }
}

void CodeGen_LLVM::visit(const Var* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Var", op->name);
  auto* v = getSymbol(op->name);
  if (v->getType()->isPointerTy()) {
    value = this->Builder->CreateLoad(v, op->name);
  } else {
    value = v;
  }
}

void CodeGen_LLVM::visit(const Neg* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Neg");
  throw logic_error("Not Implemented for Neg.");
}

void CodeGen_LLVM::visit(const Sqrt* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Sqrt");
  throw logic_error("Not Implemented for Sqrt.");
}

void CodeGen_LLVM::visit(const Add* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Add");
  auto* a = codegen(op->a);
  auto* b = codegen(op->b);
  if (op->type.isFloat()) {
    value = this->Builder->CreateFAdd(a, b);
  } else {
    value = this->Builder->CreateAdd(a, b);
  }
}

void CodeGen_LLVM::visit(const Sub* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Sub");
  throw logic_error("Not Implemented for Sub.");
}

void CodeGen_LLVM::visit(const Mul* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Mul");
  auto* a = codegen(op->a);
  auto* b = codegen(op->b);
  if (op->type.isFloat()) {
    value = this->Builder->CreateFMul(a, b);
  } else {
    value = this->Builder->CreateMul(a, b);
  }
}

void CodeGen_LLVM::visit(const Div* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Div");
  throw logic_error("Not Implemented for Div.");
}

void CodeGen_LLVM::visit(const Rem* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Rem");
  throw logic_error("Not Implemented for Rem.");
}

void CodeGen_LLVM::visit(const Min* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Min");

  if (op->operands.size() == 1) {
    value = codegen(op->operands[0]);
    return;
  }

  if (op->type.isFloat()) {
    // LLVM's minnum intrinsic only does binary ops
    value = Builder->CreateMinNum(codegen(op->operands[0]), 
                                  codegen(op->operands[1]));
    for (size_t i=2; i<op->operands.size(); i++) {
      value = Builder->CreateMinNum(value, codegen(op->operands[i]));
    }
  } else {
    llvm::PHINode* phi = nullptr;

    for (size_t i=1; i<op->operands.size(); i++) { 
      // Create the BasicBlocks
      auto* true_bb = llvm::BasicBlock::Create(*this->Context, "true_min_bb", this->Func);
      auto* false_bb = llvm::BasicBlock::Create(*this->Context, "false_min_bb", this->Func);
      auto* after_bb = llvm::BasicBlock::Create(*this->Context, "after_min_bb", this->Func);

      // Set the variables to be compared
      auto a = (phi == nullptr) ? codegen(op->operands[0]) : phi;
      auto b = codegen(op->operands[i]);

      // Create Condition and CondBranch
      auto condition = Builder->CreateICmpSLT(a, b);
      Builder->CreateCondBr(condition, true_bb, false_bb);

      // True case
      Builder->SetInsertPoint(true_bb);
      auto inst_a = llvm::dyn_cast<llvm::Instruction>(a);
      auto val_true = Builder->CreateLoad(inst_a->getOperand(0));
       Builder->CreateBr(after_bb);

      // False case
      Builder->SetInsertPoint(false_bb);
      auto inst_b = llvm::dyn_cast<llvm::Instruction>(b);
      auto val_false = Builder->CreateLoad(inst_b->getOperand(0));
    
      Builder->CreateBr(after_bb);

      // After BasicBlock
      Builder->SetInsertPoint(after_bb);

      // Create PHI node
      phi = Builder->CreatePHI(a->getType(), 2);
      phi->addIncoming(val_true, true_bb);
      phi->addIncoming(val_false, false_bb);
      value = phi; 
    }
  }
}

void CodeGen_LLVM::visit(const Max* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Max");

    if (op->operands.size() == 1) {
    value = codegen(op->operands[0]);
    return;
  }

  if (op->type.isFloat()) {
    // LLVM's minnum intrinsic only does binary ops
    value = Builder->CreateMaxNum(codegen(op->operands[0]), 
                                  codegen(op->operands[1]));
    for (size_t i=2; i<op->operands.size(); i++) {
      value = Builder->CreateMaxNum(value, codegen(op->operands[i]));
    }
  } else {
    llvm::PHINode* phi = nullptr;

    for (size_t i=1; i<op->operands.size(); i++) { 
      // Create the BasicBlocks
      auto* true_bb = llvm::BasicBlock::Create(*this->Context, "true_max_bb", this->Func);
      auto* false_bb = llvm::BasicBlock::Create(*this->Context, "false_max_bb", this->Func);
      auto* after_bb = llvm::BasicBlock::Create(*this->Context, "after_max_bb", this->Func);

      // Set the variables to be compared
      auto a = (phi == nullptr) ? codegen(op->operands[0]) : phi;
      auto b = codegen(op->operands[i]);

      // Create Condition and CondBranch
      auto condition = Builder->CreateICmpSGT(a, b);
      Builder->CreateCondBr(condition, true_bb, false_bb);

      // True case
      Builder->SetInsertPoint(true_bb);
      auto inst_a = llvm::dyn_cast<llvm::Instruction>(a);
      auto val_true = Builder->CreateLoad(inst_a->getOperand(0));
       Builder->CreateBr(after_bb);

      // False case
      Builder->SetInsertPoint(false_bb);
      auto inst_b = llvm::dyn_cast<llvm::Instruction>(b);
      auto val_false = Builder->CreateLoad(inst_b->getOperand(0));
    
      Builder->CreateBr(after_bb);

      // After BasicBlock
      Builder->SetInsertPoint(after_bb);

      // Create PHI node
      phi = Builder->CreatePHI(a->getType(), 2);
      phi->addIncoming(val_true, true_bb);
      phi->addIncoming(val_false, false_bb);
      value = phi; 
    }
  } 
}

void CodeGen_LLVM::visit(const BitAnd* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "BitAnd");
  value = Builder->CreateAnd(codegen(op->a), codegen(op->b));
}

void CodeGen_LLVM::visit(const BitOr* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "BitOr");
  value = Builder->CreateOr(codegen(op->a), codegen(op->b));
}

void CodeGen_LLVM::visit(const Eq* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Eq");
  auto* a = codegen(op->a);
  auto* b = codegen(op->b);

  if (op->type.isFloat()) {
    if (auto* FPa = llvm::dyn_cast<llvm::ConstantFP>(a)) {
      if (auto* FPb = llvm::dyn_cast<llvm::ConstantFP>(b)) {
        if (FPa->isNaN() || FPb->isNaN())
          value = Builder->CreateFCmpUEQ(a,b); // ULE means that either operand may be a QNAN.
        else
          value = Builder->CreateFCmpOEQ(a,b); // OLE means that neither operand is a QNAN
      }
    }
  } else {
    value = Builder->CreateICmpEQ(a, b);
  }
}

void CodeGen_LLVM::visit(const Neq* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Neq");
  auto* a = codegen(op->a);
  auto* b = codegen(op->b);

  if (op->type.isFloat()) {
    if (auto* FPa = llvm::dyn_cast<llvm::ConstantFP>(a)) {
      if (auto* FPb = llvm::dyn_cast<llvm::ConstantFP>(b)) {
        if (FPa->isNaN() || FPb->isNaN())
          value = Builder->CreateFCmpONE(a,b); // ULE means that either operand may be a QNAN.
        else
          value = Builder->CreateFCmpONE(a,b); // OLE means that neither operand is a QNAN
      }
    }
  } else {
    value = Builder->CreateICmpNE(a, b);
  }
}

void CodeGen_LLVM::visit(const Gt* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Gt");
  auto* a = codegen(op->a);
  auto* b = codegen(op->b);

  if (op->type.isFloat()) {
    if (auto* FPa = llvm::dyn_cast<llvm::ConstantFP>(a)) {
      if (auto* FPb = llvm::dyn_cast<llvm::ConstantFP>(b)) {
        if (FPa->isNaN() || FPb->isNaN())
          value = Builder->CreateFCmpUGT(a,b); // ULE means that either operand may be a QNAN.
        else
          value = Builder->CreateFCmpOGT(a,b); // OLE means that neither operand is a QNAN
      }
    }
  } else if (op->type.isUInt()) {
    value = Builder->CreateICmpUGT(a, b);
  } else {
    value = Builder->CreateICmpSGT(a, b);
  }
}

void CodeGen_LLVM::visit(const Lt* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Lt");
  auto* a = codegen(op->a);
  auto* b = codegen(op->b);

  if (op->type.isFloat()) {
    if (auto* FPa = llvm::dyn_cast<llvm::ConstantFP>(a)) {
      if (auto* FPb = llvm::dyn_cast<llvm::ConstantFP>(b)) {
        if (FPa->isNaN() || FPb->isNaN())
          value = Builder->CreateFCmpULT(a,b); // ULE means that either operand may be a QNAN.
        else
          value = Builder->CreateFCmpOLT(a,b); // OLE means that neither operand is a QNAN
      }
    }
  } else if (op->type.isUInt()) {
    value = Builder->CreateICmpULT(a, b);
  } else {
    value = Builder->CreateICmpSLT(a, b);
  }
}

void CodeGen_LLVM::visit(const Gte* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Gte");
  auto* a = codegen(op->a);
  auto* b = codegen(op->b);

  if (op->type.isFloat()) {
    if (auto* FPa = llvm::dyn_cast<llvm::ConstantFP>(a)) {
      if (auto* FPb = llvm::dyn_cast<llvm::ConstantFP>(b)) {
        if (FPa->isNaN() || FPb->isNaN())
          value = Builder->CreateFCmpUGE(a,b); // ULE means that either operand may be a QNAN.
        else
          value = Builder->CreateFCmpOGE(a,b); // OLE means that neither operand is a QNAN
      }
    }
  } else if (op->type.isUInt()) {
    value = Builder->CreateICmpUGE(a, b);
  } else {
    value = Builder->CreateICmpSGE(a, b);
  }
}

void CodeGen_LLVM::visit(const Lte* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Lte");
  auto* a = codegen(op->a);
  auto* b = codegen(op->b);

  if (op->type.isFloat()) {
    if (auto* FPa = llvm::dyn_cast<llvm::ConstantFP>(a)) {
      if (auto* FPb = llvm::dyn_cast<llvm::ConstantFP>(b)) {
        if (FPa->isNaN() || FPb->isNaN())
          value = Builder->CreateFCmpULE(a,b); // ULE means that either operand may be a QNAN.
        else
          value = Builder->CreateFCmpOLE(a,b); // OLE means that neither operand is a QNAN
      }
    }
  } else if (op->type.isUInt()) {
    value = Builder->CreateICmpULE(a, b);
  } else {
    value = Builder->CreateICmpSLE(a, b);
  }
}

void CodeGen_LLVM::visit(const And* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "And");
  auto* actual_bb = Builder->GetInsertBlock();

  // Create the BasicBlocks
  auto* true_bb = llvm::BasicBlock::Create(*this->Context, "true_and_bb", this->Func);
  auto* end_bb = llvm::BasicBlock::Create(*this->Context, "end_and_bb", this->Func);

  // Generate first condition
  auto a = codegen(op->a);
  Builder->CreateCondBr(a, true_bb, end_bb);

  // If true -> Generate sencond condition
  Builder->SetInsertPoint(true_bb);
  auto b = codegen(op->b);
  Builder->CreateBr(end_bb);

  // Create the PHI node on the end_bb to get the result from a && b
  Builder->SetInsertPoint(end_bb);
  auto phi = Builder->CreatePHI(get_int_type(1, *this->Context), 2);
  phi->addIncoming(a, actual_bb);
  phi->addIncoming(b, true_bb);
  value = phi;
}

void CodeGen_LLVM::visit(const Or* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Or");
  auto* actual_bb = Builder->GetInsertBlock();

  // Create the BasicBlocks
  auto* false_bb = llvm::BasicBlock::Create(*this->Context, "false_or_bb", this->Func);
  auto* end_bb = llvm::BasicBlock::Create(*this->Context, "end_or_bb", this->Func);
  
  // Generate first condition
  auto a = codegen(op->a);
  Builder->CreateCondBr(a, false_bb, end_bb);

  // If false -> Generate the sencond condition to test
  Builder->SetInsertPoint(false_bb);
  auto b = codegen(op->b);
  Builder->CreateBr(end_bb);

  // Create the PHI node on the end_bb to get the result from a || b
  Builder->SetInsertPoint(end_bb);
  auto phi = Builder->CreatePHI(get_int_type(1, *this->Context), 2);
  phi->addIncoming(a, actual_bb);
  phi->addIncoming(b, false_bb);
  value = phi;
}

void CodeGen_LLVM::visit(const Cast* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Cast");
    if (op->type.isFloat()) {
    value = Builder->CreateFPCast(codegen(op->a), llvmTypeOf(op->type));
  } else {
    value = Builder->CreateIntCast(codegen(op->a), llvmTypeOf(op->type),
            !op->type.isUInt());
  }
}

void CodeGen_LLVM::visit(const Call* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Call");
  throw logic_error("Not Implemented for Call.");
}

void CodeGen_LLVM::visit(const IfThenElse* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "IfThenElse");

  // Create the BasicBlocks
  auto* true_bb = llvm::BasicBlock::Create(*this->Context, "true_bb", this->Func);
  auto* false_bb = llvm::BasicBlock::Create(*this->Context, "false_bb", this->Func);
  auto* after_bb = llvm::BasicBlock::Create(*this->Context, "after_bb", this->Func);

  // Create condition
  Builder->CreateCondBr(codegen(op->cond), true_bb, false_bb);

  // True case
  Builder->SetInsertPoint(true_bb);
  codegen(op->then);
  Builder->CreateBr(after_bb);

  // False Case
  Builder->SetInsertPoint(false_bb);
  if (op->otherwise != nullptr) {
    codegen(op->otherwise);
  }
  Builder->CreateBr(after_bb);

  // Set the Insertion point to the next BasicBlock
  Builder->SetInsertPoint(after_bb);
}

namespace {
  Stmt caseToIfThenElse(std::vector<std::pair<Expr,Stmt>> clauses, bool alwaysMatch) {
    std::vector<std::pair<Expr,Stmt>> rest(clauses.begin()+1, clauses.end());
    if (rest.size() == 0) {
      // if alwaysMatch is true, then this one goes into the else clause,
      // otherwise, we generate an empty else clause
      return !alwaysMatch ? clauses[0].second :
        IfThenElse::make(clauses[0].first, clauses[0].second, Comment::make(""));
    } else {
      return IfThenElse::make(clauses[0].first,
                            clauses[0].second,
                            caseToIfThenElse(rest, alwaysMatch));
    }
  }
} // anonymous namespace

void CodeGen_LLVM::visit(const Case* op) { // Not tested
  auto _ = CodeGen_LLVM::IndentHelper(this, "Case");
  codegen(caseToIfThenElse(op->clauses, op->alwaysMatch));
}

void CodeGen_LLVM::visit(const Switch* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Switch");
  throw logic_error("Not Implemented for Switch.");
}

void CodeGen_LLVM::visit(const Load* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Load");

  auto* loc = codegen(op->loc);
  auto* arr = codegen(op->arr);
  auto* gep = this->Builder->CreateInBoundsGEP(arr, loc);
  value = this->Builder->CreateLoad(gep);
}

void CodeGen_LLVM::visit(const Malloc* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Malloc");
  throw logic_error("Not Implemented for Malloc.");
}

void CodeGen_LLVM::visit(const Sizeof* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Sizeof");
  throw logic_error("Not Implemented for Sizeof.");
}

void CodeGen_LLVM::visit(const Store* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Store");

  auto* loc = codegen(op->loc);
  auto* arr = codegen(op->arr);
  auto* gep = this->Builder->CreateInBoundsGEP(arr, loc);  // arr[loc]
  auto* data = codegen(op->data);                          // ... = data
  this->Builder->CreateStore(data, gep);                   // arr[loc] = data
}

void CodeGen_LLVM::visit(const For* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "For");

  auto start = codegen(op->start);
  auto end = codegen(op->end);
  taco_iassert(start->getType()->isIntegerTy());
  taco_iassert(end->getType()->isIntegerTy());

  llvm::BasicBlock* pre_header = this->Builder->GetInsertBlock();

  // Create a new basic block for the loop
  llvm::BasicBlock* header = llvm::BasicBlock::Create(*this->Context, "for_header", this->Func);

  llvm::BasicBlock* body = llvm::BasicBlock::Create(*this->Context, "for_body", this->Func);

  llvm::BasicBlock* latch = llvm::BasicBlock::Create(*this->Context, "for_latch", this->Func);

  llvm::BasicBlock* exit = llvm::BasicBlock::Create(*this->Context, "for_exit", this->Func);

  this->Builder->CreateBr(header);  // pre-header -> header

  this->Builder->SetInsertPoint(header);

  // Initialize header with PHI node
  const Var* var = op->var.as<Var>();
  auto phi = this->Builder->CreatePHI(start->getType(), 2 /* num values */, var->name);
  pushSymbol(var->name, phi);

  // Compute exit condition
  auto cond = this->Builder->CreateICmpSLT(phi, end); // Always less then?
  this->Builder->CreateCondBr(cond, body, exit);

  // Compute increment and jump back to header
  this->Builder->SetInsertPoint(latch);
  auto incr = this->Builder->CreateAdd(phi, codegen(op->increment));
  this->Builder->CreateBr(header);  // latch -> header

  // Add values to the PHI node
  phi->addIncoming(start, pre_header);
  phi->addIncoming(incr, latch);

  // Connect body to latch
  this->Builder->SetInsertPoint(body);
  op->contents.accept(this);
  this->Builder->CreateBr(latch);  // body -> latch

  this->Builder->SetInsertPoint(exit);
  removeSymbol(var->name);
}

void CodeGen_LLVM::visit(const While* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "While");
  taco_tassert(op->kind == LoopKind::Serial) <<
    "Only serial loop codegen supported by LLVM backend";

  // Create a new basic block for the loop
  llvm::BasicBlock* header = llvm::BasicBlock::Create(*this->Context, "while_header", this->Func);
  llvm::BasicBlock* loop = llvm::BasicBlock::Create(*this->Context, "while_body", this->Func);
  llvm::BasicBlock* exit = llvm::BasicBlock::Create(*this->Context, "while_end", this->Func);
  
  // pre_header -> header
  Builder->CreateBr(header);

  // entry condition
  Builder->SetInsertPoint(header);
  auto condition = codegen(op->cond);
  Builder->CreateCondBr(condition, loop, exit);

  // codegen body
  Builder->SetInsertPoint(loop);
  codegen(op->contents);

  // create unconditional branch to check
  Builder->CreateBr(header);

  // seet the insert point for the exit loop
  Builder->SetInsertPoint(exit);
}

void CodeGen_LLVM::visit(const Block* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Block");
  for (const auto& s : op->contents) {
    s.accept(this);
  }
}

void CodeGen_LLVM::visit(const Scope* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Scope");
  pushScope();
  op->scopedStmt.accept(this);
  popScope();
}

void CodeGen_LLVM::init_codegen() {
  if (this->Module == nullptr) {
    this->Context = std::make_unique<llvm::LLVMContext>();
    this->Module = std::make_unique<llvm::Module>("taco_module", *this->Context);
    this->Builder = std::make_unique<llvm::IRBuilder<>>(*this->Context);

    auto i32 = llvm::Type::getInt32Ty(*this->Context);
    auto i32p = i32->getPointerTo();

    auto u8 = llvm::Type::getInt8Ty(*this->Context);
    auto u8p = u8->getPointerTo();
    auto u8ppp = u8->getPointerTo()->getPointerTo()->getPointerTo();

    /* See file include/taco/taco_tensor_t.h for the struct tensor definition */
    this->tensorStruct = llvm::StructType::create(*this->Context,
                                                  {
                                                      i32,   /* order */
                                                      i32p,  /* dimension */
                                                      i32,   /* csize */
                                                      i32p,  /* mode_ordering */
                                                      i32p,  /* mode_types */
                                                      u8ppp, /* indices */
                                                      u8p,   /* vals */
                                                      i32,   /* vals_size */
                                                  },
                                                  "tensorStruct");
    this->tensorStructPtr = this->tensorStruct->getPointerTo();
  }
}

void CodeGen_LLVM::visit(const Function* func) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Function @" + func->name);

  /*
    This method creates a function. By calling convention, the function
    returns 0 on success or 1 otherwise.
  */

  // 1. find the arguments to @func
  FindVars varFinder(func->inputs, func->outputs, this);

  // 2. get the arguments types
  // Are all arguments tensors?

  // 3. convert the types to the LLVM correspondent ones
  int n_args = func->inputs.size() + func->outputs.size();
  std::vector<llvm::Type*> args;
  for (int i = 0; i < n_args; i++) {
    args.push_back(this->tensorStructPtr);
  }
  auto i32 = llvm::Type::getInt32Ty(*this->Context);

  // 4. create a new function in the module with the given types
  this->Func = llvm::Function::Create(llvm::FunctionType::get(i32, args, false),
                                      llvm::GlobalValue::ExternalLinkage,
                                      func->name,
                                      this->Module.get());

  // 5. Create the first basic block
  this->Builder->SetInsertPoint(llvm::BasicBlock::Create(*this->Context, "entry", this->Func));

  // 6. Push arguments to symbol table
  pushScope();
  size_t idx = 0;

  for (llvm::Argument& arg : this->Func->args()) {
    // output comes first
    const auto* var = idx < func->outputs.size()
                          ? func->outputs[idx++].as<Var>()
                          : func->inputs[(idx++ - func->outputs.size())].as<Var>();
    // set arg name
    arg.setName(var->name);

    // set arg flags
    arg.addAttr(llvm::Attribute::NoCapture);

    // 6.1 push args to symbol table
    pushSymbol(var->name, &arg);
  }

  // 7. visit function body
  func->body.accept(this);

  // 8. Create an exit basic block and exit it
  llvm::BasicBlock* exit = llvm::BasicBlock::Create(*this->Context, "exit", this->Func);
  this->Builder->CreateBr(exit);
  this->Builder->SetInsertPoint(exit);                       // ... -> exit
  this->Builder->CreateRet(llvm::ConstantInt::get(i32, 0));  // return 0

  // 9. Verify the created module
  llvm::verifyModule(*this->Module, &llvm::errs());

  // llvm::outs() << *this->Module << "\n";
}

void CodeGen_LLVM::visit(const VarDecl* op) {
  const Var* lhs = op->var.as<Var>();
  auto _ = CodeGen_LLVM::IndentHelper(this, "VarDecl", lhs->name);
  // Create the pointer
  llvm::Type* rhs_llvm_type = llvmTypeOf(op->rhs.type());
  auto* ptr = this->Builder->CreateAlloca(rhs_llvm_type);

  // visit op rhs to produce a value
  // codegen ensures that a LLVM value was produced
  this->Builder->CreateStore(codegen(op->rhs), ptr);

  // Store the symbol/ptr in the symbol table
  pushSymbol(lhs->name, ptr);
}

void CodeGen_LLVM::visit(const Assign* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Assign");

  const Var* lhs = op->lhs.as<Var>();
  auto *rhs = codegen(op->rhs);
  auto *ptr = getSymbol(lhs->name);
  this->Builder->CreateStore(rhs, ptr);
}

void CodeGen_LLVM::visit(const Yield* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Yield");
  throw logic_error("Not Implemented for Yield.");
}

void CodeGen_LLVM::visit(const Allocate* op) {
auto _ = CodeGen_LLVM::IndentHelper(this, "Warning: Missing Allocate implementation");
/*
  auto _ = CodeGen_LLVM::IndentHelper(this, "Allocate");
  auto voidptr = get_void_ptr_type(*this->Context);
  auto i64 = get_int_type(64, *this->Context);
  auto i32p = get_int_ptr_type(32, *this->Context);

  auto alloca = this->Builder->CreateAlloca(i32p);
//  auto var = codegen(op->var);
  auto num_elements = codegen(op->num_elements);
  auto sext_num_elements = this->Builder->CreateSExt(num_elements, i64);
  
  if (op->is_realloc) { 
    auto size = this->Builder->CreateMul(
        sext_num_elements, get_int_constant(64, 4, *this->Context), "realloc.size");

    auto ret = emitExternalCall("realloc", voidptr, {i64}, {size});
    ret->setName("realloc.ret");
    
    auto bitcast = this->Builder->CreateBitCast(ret, i32p);
    this->Builder->CreateStore(bitcast, alloca);
  } else {
    auto size = this->Builder->CreateMul(
      sext_num_elements, get_int_constant(64, 4, *this->Context));
      
   auto call = emitExternalCall("malloc", voidptr, {i64}, {size});
   call->setName("calloc.ret");

   auto bitcast = this->Builder->CreateBitCast(call, i32p);
   this->Builder->CreateStore(bitcast, alloca);
  }*/
}

void CodeGen_LLVM::visit(const Free* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Free");
  throw logic_error("Not Implemented for Free.");
}

void CodeGen_LLVM::visit(const Comment* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Comment");
  // no op, do nothing
}

void CodeGen_LLVM::visit(const BlankLine *op) {
// no-op, do nothing
}

void CodeGen_LLVM::visit(const Break* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Break");
  throw logic_error("Not Implemented for Break.");
}

void CodeGen_LLVM::visit(const Print* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "Print");
  throw logic_error("Not Implemented for Print.");
}

std::string CodeGen_LLVM::tensorPropertyToString(const TensorProperty t) {
  switch (t) {
    case TensorProperty::Order:
      return "Order";
    case TensorProperty::Dimension:
      return "Dimension";
    case TensorProperty::ComponentSize:
      return "ComponentSize";
    case TensorProperty::ModeOrdering:
      return "ModeOrdering";
    case TensorProperty::ModeTypes:
      return "ModeTypes";
    case TensorProperty::Indices:
      return "Indices";
    case TensorProperty::Values:
      return "Values";
    case TensorProperty::ValuesSize:
      return "ValuesSize";
    default:
      break;
  }
  taco_unreachable;
  return "";
}

void CodeGen_LLVM::visit(const GetProperty* op) {
  auto _ = CodeGen_LLVM::IndentHelper(this, "GetProperty");

  const std::string& name = op->tensor.as<Var>()->name;
  llvm::Value* tensor = getSymbol(name);

  auto* tensorType_pp = llvmTypeOf(op->type)->getPointerTo()->getPointerTo();  // TensorType**
  auto* i32p = get_int_ptr_type(32, *this->Context);

  switch (op->property) {
    case TensorProperty::Dimension: {
      auto* dim = this->Builder->CreateStructGEP(
          tensor, (int) TensorProperty::Dimension, name + ".gep.dim");
      value = this->Builder->CreateLoad(this->Builder->CreateLoad(dim), name + ".dim");
      break;
    }
    case TensorProperty::Values: {
      auto* gep = this->Builder->CreateStructGEP(tensor, (int) TensorProperty::Values);
      auto* bitcast = this->Builder->CreateBitCast(gep, tensorType_pp);
      value = this->Builder->CreateLoad(bitcast, name + ".vals");
      break;
    }
    case TensorProperty::Order:
    case TensorProperty::ComponentSize:
    case TensorProperty::ModeOrdering:
    case TensorProperty::ModeTypes:
    case TensorProperty::Indices: {
      auto* mode = get_int_constant(32, op->mode, *this->Context);
      auto* index = get_int_constant(32, op->index, *this->Context);

      auto* load1 = this->Builder->CreateLoad(this->Builder->CreateStructGEP(tensor, (int)TensorProperty::Indices));  // i8***
      auto* load2 = this->Builder->CreateLoad(this->Builder->CreateInBoundsGEP(load1, mode));  // i8**
      auto* load3 = this->Builder->CreateLoad(this->Builder->CreateInBoundsGEP(load2, index));  // i8*
      value = this->Builder->CreateBitCast(load3, i32p);
      break;
    }
    case TensorProperty::ValuesSize:
    default:
      throw logic_error("GetProperty not implemented for " + tensorPropertyToString(op->property));
  }
}

}  // namespace ir
}  // namespace taco
#endif  // HAVE_LLVM
