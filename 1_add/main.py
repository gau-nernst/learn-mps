import torch
import torch.utils.cpp_extension

module = torch.utils.cpp_extension.load(
    "module",
    sources=["add.mm"],
    verbose=True,
)

A = torch.ones(1024, device="mps")
B = torch.ones(1024, device="mps") * 2
C = module.add(A, B)

print(f"{A=}")
print(f"{B=}")
print(f"{C=}")
