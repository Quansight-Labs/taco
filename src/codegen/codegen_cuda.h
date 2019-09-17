#ifndef TACO_BACKEND_CUDA_H
#define TACO_BACKEND_CUDA_H

#include <map>
#include <vector>

#include "taco/ir/ir.h"
#include "taco/ir/ir_printer.h"
#include "codegen.h"

namespace taco {
namespace ir {


class CodeGen_CUDA : public CodeGen {
public:
  /// Initialize a code generator that generates code to an
  /// output stream.
  CodeGen_CUDA(std::ostream &dest, OutputKind outputKind);
  ~CodeGen_CUDA();

  /// Compile a lowered function
  void compile(Stmt stmt, bool isFirst=false);
  
  /// Generate shims that unpack an array of pointers representing
  /// a mix of taco_tensor_t* and scalars into a function call
  static void generateShim(const Stmt& func, std::stringstream &ret);
protected:
  using IRPrinter::visit;
  void visit(const Function*);
  void visit(const Var*);
  void visit(const For*);
  void visit(const While*);
  void visit(const GetProperty*);
  void visit(const Min*);
  void visit(const Max*);
  void visit(const Allocate*);
  void visit(const Sqrt*);
  void visit(const Add*);
  void visit(const Sub*);
  void visit(const Mul*);
  void visit(const Div*);
  void visit(const VarDecl*);
  void visit(const Literal*);
  void visit(const Yield*);
  void visit(const Call*);
  std::string printDeviceFuncName(const std::vector<std::pair<std::string, Expr>> currentParameters, int index);
  void printDeviceFuncCall(const std::vector<std::pair<std::string, Expr>> currentParameters, int index, Expr start, Expr end, Expr increment);
  void printThreadIDVariable(std::pair<std::string, Expr> threadIDVar, Expr start, Expr increment);
  void printThreadBoundCheck(std::pair<std::string, Expr> threadIDVar, Expr end);
  void printDeviceFunctions(const Function* func);
  void printBinCastedOp(Expr a, Expr b, std::string op, Precedence precedence);
  Stmt simplifyFunctionBodies(Stmt stmt);

  std::map<Expr, std::string, ExprCompare> varMap;
  std::vector<Expr> localVars;

  std::vector<std::vector<std::pair<std::string, Expr>>> deviceFunctionParameters;
  std::vector<Stmt> deviceFunctions; // expressions to replace to calls of device function

  std::ostream &out;
  
  OutputKind outputKind;

  std::string funcName;
  int labelCount;
  bool emittingCoroutine;

  class FindVars;
  class DeviceFunctionCollector;

private:
  virtual std::string restrictKeyword() const { return "__restrict__"; }
};

} // namespace ir
} // namespace taco
#endif
