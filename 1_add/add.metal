#include <metal_stdlib>
using namespace metal;

// "constant" and "device" are address spaces in metal
// [[ ]] is C++ attribute syntax
kernel void add(constant float *A [[ buffer(0) ]],
                constant float *B [[ buffer(1) ]],
                device   float *C [[ buffer(2) ]],
                uint tid [[ thread_position_in_grid ]])
{
    // we don't need to check for bounds, since we can set total number of threads exactly.
    // (compared to CUDA, which the number of threads = thread block size * number of thread blocks).
    // it's also referred to as "nonuniform threadgroup sizes" in Metal.
    C[tid] = A[tid] + B[tid];
}
