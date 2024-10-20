# Add kernel

Resources:
- https://www.youtube.com/watch?v=VQK28rRK6OU and https://github.com/twohyjr/Metal-Game-Engine-Tutorial/tree/master/MetalCode/AddCompute/AddCompute
- https://developer.apple.com/documentation/metal/performing_calculations_on_a_gpu?language=objc

Objective-C resources:
- https://www.hackingwithswift.com/articles/114/objective-c-to-swift-conversion-cheat-sheet
- https://en.wikipedia.org/wiki/Objective-C
- https://developer.apple.com/library/archive/documentation/Cocoa/Conceptual/ProgrammingWithObjectiveC/Introduction/Introduction.html

```bash
xcrun metal *.metal
clang++ -std=c++17 main.mm -o main -framework Metal -framework Foundation
./main
```

Learnings:
- Set up Objective-C language server: https://github.com/MaskRay/ccls (can be installed with homebrew).
- Compile Metal shaders with `xcrun metal *.metal`. This will produce `default.metallib`. In host code, load this default library with `id<MTLLibrary> lib = [device newDefaultLibrary]`.
- Metal shader concepts: Compute pipeline, command queue, command buffer, command encoder, grid, thread group.
- Objective-C has Automatic Reference Counting (ARC): it will deallocate objects automatically once there are no more references to it. If we return a buffer owned by an Objective-C object to C++ or Python, we can do `[obj retain]` to manually increase reference count (so it won't get deallocated), but we have to do `[obj release]` later to avoid memory leak.
- PyTorch MPS stream is a tuple of `MTLCommandQueue` and `dispatch_queue_t`. We use these (instead of manually creating our own) when integrating custom op in PyTorch.
