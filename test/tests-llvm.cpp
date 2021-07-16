#include "test.h"
#include "test_tensors.h"
#include "taco/tensor.h"
#include "taco/llvm.h"

using namespace taco;

const IndexVar i("i"), j("j"), k("k");

TEST(llvm, add)
{
  Tensor<int32_t> A("A", {3}, Format({Dense}, {0}));
  Tensor<int32_t> B("B", {3}, Format({Dense}, {0}));
  Tensor<int32_t> C("C", {3}, Format({Dense}, {0}));
  Tensor<int32_t> E("E", {3}, Format({Dense}, {0}));

  E.insert({0}, (int32_t)5);
  E.insert({1}, (int32_t)7);
  E.insert({2}, (int32_t)9);

  B.insert({0}, (int32_t)1);
  B.insert({1}, (int32_t)2);
  B.insert({2}, (int32_t)3);

  C.insert({0}, (int32_t)4);
  C.insert({1}, (int32_t)5);
  C.insert({2}, (int32_t)6);

  // Pack inserted data as described by the formats
  B.pack();
  C.pack();

  // Form an expression
  A(i) = B(i) + C(i);

  set_LLVM_codegen_enabled(true);
  A.evaluate();

  ASSERT_TRUE(equals(E, A));
}
