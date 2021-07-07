#include <iostream>
#include "include/taco.h"
#include "include/taco/llvm.h"

using namespace taco;

int main()
{
    // Create formats
    Format list({Dense});

    // Create tensors
    Tensor<double> A({3}, list);
    Tensor<double> B({3}, list);

    // Insert data into B
    B.insert({0}, 1.0);
    B.insert({1}, 2.0);
    B.insert({1}, 3.0);

    // Pack inserted data as described by the formats
    B.pack();

    // Form a tensor-vector multiplication expression
    IndexVar i;
    A(i) = B(i);

    // Compile the expression
    set_LLVM_codegen_enabled(true);
    A.compile();

    // Assemble A's indices and numerically compute the result
    A.assemble();
    A.compute();

    std::cout << A << std::endl;

    return 0;
}
