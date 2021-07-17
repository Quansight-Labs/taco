#include <iostream>
#include "../include/taco.h"
#include "../include/taco/llvm.h"

using namespace taco;

int main() {
    // Create formats
    Format list({Sparse});

    // Create tensofloatr
    Tensor<float> A({3}, list);
    Tensor<float> B({3}, list);
    Tensor<float> C({3}, list);

    // Insert data into B
    B.insert({0}, (float) 1.5);
    B.insert({1}, (float) 2.5);
    B.insert({2}, (float) 3.5);

    // Insert data into C
    C.insert({0}, (float) 4.5);
    C.insert({1}, (float) 5.5);
    C.insert({2}, (float) 6.5);


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

