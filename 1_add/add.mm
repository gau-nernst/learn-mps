#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <Metal/Metal.h>

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <stdlib.h>


// https://github.com/malfet/llm_experiments/blob/main/metal-perf/int8mm.mm
static id<MTLLibrary> lib_from_file(id<MTLDevice> device, const std::string &filepath) {
  std::ifstream ifs(filepath);
  std::stringstream ss;
  ss << ifs.rdbuf();
  ifs.close();

  NSError *error = nil;
  id<MTLLibrary> lib = [device
    newLibraryWithSource:[NSString stringWithUTF8String:ss.str().c_str()]
    options:nil
    error:&error];
  TORCH_CHECK(lib != nil, "Failed to compile ", filepath, ". Error: ", error.localizedDescription.UTF8String);
  return lib;
}

// https://github.com/pytorch/pytorch/blob/v2.5.0/aten/src/ATen/native/mps/OperationUtils.h#L96-L98
static id<MTLBuffer> getMTLBufferStorage(const at::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
static id<MTLLibrary> lib = lib_from_file(device, "add.metal");


torch::Tensor add(torch::Tensor A, torch::Tensor B) {
  int N = A.numel();
  TORCH_CHECK(B.numel() == N, "A and B must have the same size");
  TORCH_CHECK(A.is_mps() && B.is_mps(), "A and B must be MPS");
  torch::Tensor C = torch::empty(N, A.options());

  id<MTLFunction> addFunc = [lib newFunctionWithName:@"add"];
  TORCH_CHECK(addFunc != nil, "Failed to find add function")
  NSError *error = nil;
  id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:addFunc error:&error];

  id<MTLBuffer> A_buf = getMTLBufferStorage(A);
  id<MTLBuffer> B_buf = getMTLBufferStorage(B);
  id<MTLBuffer> C_buf = getMTLBufferStorage(C);

  // in Pytorch, MPS stream = (MTLCommandQueue, dispatch_queue_t)
  // MTLCommandQueue is Metal thing, while dispatch_queue_t is Objective-C thing
  // https://github.com/pytorch/pytorch/blob/v2.5.0/aten/src/ATen/mps/MPSStream.mm
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  id<MTLCommandBuffer> command_buf = stream->commandBuffer();
  dispatch_queue_t dispatch_queue = stream->queue();

  // this is Objective-C thing
  dispatch_sync(dispatch_queue, ^() {
    id<MTLComputeCommandEncoder> compute_enc = [command_buf computeCommandEncoder];
    [compute_enc setComputePipelineState:pso];
    [compute_enc setBuffer:A_buf offset:A.storage_offset() * A.element_size() atIndex:0];
    [compute_enc setBuffer:B_buf offset:B.storage_offset() * B.element_size() atIndex:1];
    [compute_enc setBuffer:C_buf offset:C.storage_offset() * C.element_size() atIndex:2];

    MTLSize grid_size = MTLSizeMake(N, 1, 1);
    MTLSize group_size = MTLSizeMake(MIN(256, N), 1, 1);
    [compute_enc dispatchThreads:grid_size threadsPerThreadgroup:group_size];
    [compute_enc endEncoding];
    stream->synchronize(at::mps::SyncType::COMMIT);  // call this instead of [command_buff commit]
  });

  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", &add, "Add two vectors");
}
