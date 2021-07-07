#include <iostream>
#include "include/taco.h"
#include "include/taco/llvm.h"

using namespace taco;

int main()
{
    // Create formats
    Format list({Dense});

    // Create tensors
    Tensor<int32_t> A({3}, list);
    Tensor<int32_t> B({3}, list);
    Tensor<int32_t> C({3}, list);

    // Insert data into B
    B.insert({0}, (int32_t) 1);
    B.insert({1}, (int32_t) 2);
    B.insert({2}, (int32_t) 3);

    C.insert({0}, (int32_t) 4);
    C.insert({1}, (int32_t) 5);
    C.insert({2}, (int32_t) 6);

    // Pack inserted data as described by the formats
    B.pack();
    C.pack();

    // Form a tensor-vector multiplication expression
    IndexVar i;
    A(i) = B(i) + C(i);

    // Compile the expression
    set_LLVM_codegen_enabled(true);
    A.compile();

    // Assemble A's indices and numerically compute the result
    A.assemble();
    A.compute();

    std::cout << A << std::endl;

    return 0;
}
