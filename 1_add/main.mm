#include <Metal/Metal.h>

float add(float a, float b) { return a + b; }

int main() {
  NSString *message = @"A string in Objective-C";
  NSLog(@"Hello, %@", message);

  // it's possible to mix C/C++ with Objective-C
  // %@ format specifier is for id<>
  float a = 0.0f;
  float b = 1.0f;
  NSLog(@"Result: %f", add(a, b));

  // id<> is a pointer to an Objective-C object
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  NSLog(@"Device: %@", device);

  // call method "newDefaultLibrary" on "device" object with no arguments
  id<MTLLibrary> lib = [device newDefaultLibrary];
  NSLog(@"Library: %@", lib);

  // call method with 1 argument
  id<MTLFunction> addFunc = [lib newFunctionWithName:@"add"];
  NSLog(@"Function: %@", addFunc);

  // call method with 2 arguments. the 2nd argument has label "error"
  // pipeline state object
  NSError *error = nil;
  id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:addFunc error:&error];

  // allocate memory that GPU can access
  const int N = 1024;
  const int num_bytes = N * sizeof(float);
  id<MTLBuffer> A = [device newBufferWithLength:num_bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> B = [device newBufferWithLength:num_bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> C = [device newBufferWithLength:num_bytes options:MTLResourceStorageModeShared];

  // fill data
  float *A_ptr = reinterpret_cast<float *>(A.contents);
  float *B_ptr = reinterpret_cast<float *>(B.contents);
  for (int i = 0; i < N; i++) {
    A_ptr[i] = 1.0f;
    B_ptr[i] = 2.0f;
  }

  id<MTLCommandQueue> queue = [device newCommandQueue];
  id<MTLCommandBuffer> command_buf = [queue commandBuffer];
  id<MTLComputeCommandEncoder> compute_enc = [command_buf computeCommandEncoder];
  [compute_enc setComputePipelineState:pso];

  // this corresponds to what we write in our shader. A is buffer 0, B is buffer 1, and so on.
  [compute_enc setBuffer:A offset:0 atIndex:0];
  [compute_enc setBuffer:B offset:0 atIndex:1];
  [compute_enc setBuffer:C offset:0 atIndex:2];

  // launch params
  NSUInteger max_thread_group_size = pso.maxTotalThreadsPerThreadgroup;  // corresponds to CUDA's thread block size
  NSLog(@"Max thread group size: %tu", max_thread_group_size);

  // MTLSize corresponds to CUDA's dim3
  MTLSize grid_size = MTLSizeMake(N, 1, 1);  // problem size. this is different from CUDA's grid size, which is the number of thread blocks.
  MTLSize group_size = MTLSizeMake(MIN(max_thread_group_size, N), 1, 1);
  [compute_enc dispatchThreads:grid_size threadsPerThreadgroup:group_size];  // no need bounds check
  // [compute_enc dispatchThreadgroups:num_groups threadsPerThreadgroup:group_size];  // closer to CUDA. needs bounds check

  [compute_enc endEncoding];         // finish encoding commands to the compute pass
  [command_buf commit];              // place the commands to queue
  [command_buf waitUntilCompleted];  // synchronize

  // read results
  float *C_ptr = reinterpret_cast<float *>(C.contents);
  for (int i = 0; i < N; i++) {
    assert(C_ptr[i] == 3.0f);
  }

  return 0;
}
