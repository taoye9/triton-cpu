Create a triton IR to run: lhs * rhs matmul with
    * bias
    * clamp 
packing:
    * lhs packing
    * rhs packing with bias

triton front end:
    * a mlir builder: python/src/ir.cc
    * build a mlir op with matmul with clamp and bias

mlir compiler:
* Dialect: packing, matmul MLIR in include/triton/Dialect/TritonCPU/IR/TritonCPUOps.td.
    * test/TritonCPU/dot-to-aarch64-microkernel.mlir
* pm: third_party/cpu/backend/compiler.py
    * third_party/cpu/include/TritonToTritonCPU/Passes.td
* Pass:  mlir::triton::cpu::createConvertDotToOneDNN()
    * UkernelOpsToAArch64 third_party/cpu/include/TritonCPUToLLVM/Passes.td
* libs:
    * third_party/cpu/lib/TritonCPUToLLVM/OneDNNOpsToLLVM.cpp

runtime:
* install kleidiai in third_party/cpu/TestOneDNNukernel.cpp
* install openblas

Arm kleidiai: https://github.com/ARM-software/kleidiai
Openblas: 



# environment builtup

install pytorch on neoverse-v2: 

```
wget https://artifactory.arm.com:443/artifactory/oncpuml.pypi-federated/torch/2.8.0.dev20250403/torch-2.8.0.dev20250403-cp310-cp310-manylinux_2_28_aarch64.whl
```


dsf 

sad 
