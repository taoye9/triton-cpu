# Project Tasks (Working List)
Goal: Extend Triton CPU backend with a matmul that supports packing, bias, and clamp.
- Build custom MLIR op for fused matmul + bias + clamp (packing-aware)
- Add dialect pieces (packing + fused matmul) in `include/triton/Dialect/TritonCPU/IR/TritonCPUOps.td`
- Create lowering tests (e.g. extend `test/TritonCPU/dot-to-aarch64-microkernel.mlir`)
- Implement passes in `third_party/cpu/include/TritonToTritonCPU/Passes.td`
- Integrate ukernel conversion (OneDNN / XSMM) + AArch64 path
- Link required external libs (OneDNN, XSMM, SLEEF, optional KleidiAI, OpenBLAS)

Reference libs / external acceleration:
- KleidiAI (Arm): https://github.com/ARM-software/kleidiai
- OpenBLAS: https://www.openblas.net/

Key primitives to leverage:
- `tl.dot(a, b, out_dtype=..., allow_tf32=...)` – matrix multiply intrinsic lowered through CPU pass pipeline


# Add rhs pack function support

## Python interface:

goal is to add packing function which supports the following ukernels defined in https://github.com/ARM-software/kleidiai/blob/main/kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.h

### C++ primitive interface

main packing function to be compiled from triton through mlir to c++:

```c++
/// Runs the RHS packing function for matrix multiplication.
///
/// The pointer of each buffers (RHS, bias and packed RHS) needs to be added with offset
/// calculated using the following functions:
///
///   * RHS: @ref kai_get_rhs_offset_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.
///   * Bias: @ref kai_get_packed_rhs_offset_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.
///   * Output: @ref kai_get_dst_offset_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.
///
/// @param[in] num_groups Number of groups. It must be 1.
/// @param[in] n Number of columns of the output matrix.
/// @param[in] k Common dimension between the LHS and RHS matrix.
/// @param[in] nr Block size in N dimension. It must be 16.
/// @param[in] kr Block size in K dimension. It must be 1.
/// @param[in] sr Number of kr splits. It must be 1.
/// @param[in] rhs_stride Row stride in bytes of the RHS matrix.
/// @param[in] rhs RHS matrix data buffer.
/// @param[in] bias Bias matrix data buffer.
/// @param[in] scale Scale data buffer. It must be NULL.
/// @param[out] rhs_packed Packed RHS matrix.
/// @param[in] extra_bytes Extra bytes to append to the end of each row of the packed RHS matrix. It must be 0.
/// @param[in] params Extra packing parameters. It must be NULL.
void kai_run_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t rhs_stride, const void* rhs,
    const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes, const void* params);
```


some helper functions to be used during compile-time (i.e. directly through python binding without mlir compilation):

```c++
/// Gets the size in bytes of the packed RHS buffer.
///
/// @param[in] n Number of rows.
/// @param[in] k Number of columns.
///
/// @return The size in bytes of the packed RHS buffer.
size_t kai_get_rhs_packed_size_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(size_t n, size_t k);
```

### python interface declaration

our triton kernel packs a partial B matrix tile (tileK x tileN) into a packed format using the above C++ primitive interface.
```python
# example usage
rhs_packed = run_rhs_pack(rhs_blk, bias_block)
# out: packed_rhs of shape (BLOCK_SIZE_K * BLOCK_SIZE_N)
```

in ir.cc, bind the `create_rhs_pack` to python interface through pybind11. it's a simple wrapper around mlir builder which creates an op. 

implment python wrapper functions in `core.py` and `semantics.py` to call the above C++ primitive through triton cpu backend.


### Code generation: From python AST to mlir TTIR
1. Decorate with `@triton.jit`: a `JITFunction` object captures the Python function, its signature, and constexpr annotations.
2. Indexing `fn[grid]` calls `JITFunction.__getitem__` returning a lightweight callable that stores the launch `grid`.
3. Calling that callable collects runtime args + keyword launch options (e.g. `num_threads`) and invokes `JITFunction.run`.
4. `run` builds a cache key (function source hash + constexpr values + CPUOptions hash) and looks up/creates a `CompiledKernel` via `compile()` in `python/triton/compiler/compiler.py`.
5. `compile()` drives a backend object (`CPUBackend`) which returns a `CompiledKernel` containing metadata plus a ready-to-use launcher handle supplied by `python/triton/backends/cpu/driver.py`.
6. The driver module’s generated C++ launcher (built on demand) executes an OpenMP loop over the flattened grid calling the native kernel entrypoint.

Result: Python invocation reuses cached native code, minimizing recompilation and providing a simple `kernel[grid](...)` API.

question:  tensor class methods are stub or placeholder with ... (an ellipsis literal). there are mutiple implementation. 
how are they binded during compile?
  -	Stub files (.pyi files), where the implementation exists elsewhere (e.g., in C/C++ extensions or other Python modules).
	-	Generated interfaces for native bindings (e.g., functions implemented in C/C++ via CPython API, cython, pybind11, etc.).

## MLIR Lowering

### MLIR Ops: 

#### concepts:

MLIR Op = Traits + Interfaces + Attributes + Regions + Types + Semantics

An MLIR operation (func.func, arith.addi, your custom op) is described by:
    -   Traits → opt-in compile-time behavior (“this op has X property”).
        - used in verification, transformations, pattern matching.
        - You cannot configure traits at runtime. They are part of the op class, not the op instance.
            - `if (op->hasTrait<OpTrait::ConstantLike>()) { ... }`
    -   Interfaces → virtual API that passes/rewriters can use generically
    -   Attributes → user-specified metadata stored directly on the op
    -   Operands → SSA inputs
    -   Results → SSA outputs
    -   Regions / blocks → structure
    -   Verifier + semantic rules


# Environment Setup

1. Clone & init submodules + venv:
```bash
# git clone forked repo
git clone git@github.com:taoye9/triton-cpu.git
cd triton-cpu
git submodule update --init --recursive

python -m venv .venv --prompt triton
source .venv/bin/activate
```

2. Install PyTorch (CPU):

```bash
pip3 install numpy torch --index-url https://download.pytorch.org/whl/cpu
```

3. Build Triton (editable):

```bash
export TRITON_CPU_BACKEND=1 #use the CPU backend over a GPU backend
export TRITON_BUILD_WITH_CLANG_LLD=true # use clang and lld. lld in particular results in faster builds.
export TRITON_BUILD_WITH_CCACHE=true

# change the location of the .triton directory where Triton's cache is located and downloads are stored during the build. By default, this is the user's home directory. It can be changed anytime.
export TRITON_HOME=$(pwd)
export MAX_JOBS=16 # Limit parallel jobs to reduce memory use and avoid OOMs

pip install -r python/requirements.txt # build-time dependencies
pip install --no-build-isolation -e python -v

# Optional, package build into a wheel to install on other machines.
python setup.py bdist_wheel
ls dist  # Wheel should be output in this directory
```
Notes:
- `TRITON_HOME` at repo root keeps caches local.
- `--no-build-isolation` reuses venv deps (faster rebuild cycles).


## VS Code Workspace Settings 

- Purpose: make VS Code use the project venv, provide build/test tasks, and enable debugging for Python and C++.

- Common workspace files:
    - `.vscode/settings.json` — interpreter, CMake build dir, terminal env.
    - `.vscode/tasks.json` — `CMake Configure`, `CMake Build`, `Run Pytest` tasks.
    - `.vscode/launch.json` — Python and C++ debug configs.

- `.vscode/settings.json`:

    ```json
    {
        "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
        "python.terminal.activateEnvironment": true,
        "cmake.buildDirectory": "${workspaceFolder}/build",
        "terminal.integrated.env.linux": {
            "PYTHONNOUSERSITE": "1",
            "PATH": "${workspaceFolder}/.venv/bin:${env:PATH}",
            "LD_LIBRARY_PATH": "${workspaceFolder}/build:${env:LD_LIBRARY_PATH}"
        }
    }
    ```

### Code Intelligence

- c++ code:
Provide the absolute path to `compile_commands.json` in C/C++ configuration for proper symbol navigation.
```bash
find python/build -name 'compile_commands.json' | xargs readlink -f
# example:
# python/build/cmake.linux-aarch64-cpython-3.12/compile_commands.json
```

- **MLIR code:**

VS Code MLIR settings requires point the MLIR language servers to the correct TableGen server binary with proper include paths.

```bash
# find the tablegen compilation database
find . -type f -name 'tablegen_compile_commands.yml'

# find MLIR/TableGen LSP server binaries
find . -type f -executable \( -name 'mlir-lsp-server' -o -name 'mlir-pdll-lsp-server' -o -name 'tblgen-lsp-server' \)
```

in `.vscode/settings.json`, add the above abs path in following setting (adjust the path as necessary):

```json
{
    "mlir.onSettingsChanged": "restart",
    "mlir.server_path": "...",
    "mlir.pdll_server_path": ...",
    "mlir.tablegen_server_path": "...",
    "mlir.tablegen_compilation_databases": [
        "..."
    ],
}
```

- MLIR tools:
utiles mlir-opt, mlir-translate, etc. from the build. Add the build bin dir to PATH in terminal env settings.

```json
{
  "terminal.integrated.env.linux": {
    "PATH": "${workspaceFolder}/triton-cpu/python/build/cmake.linux-aarch64-cpython-3.12/bin/:${env:PATH}"
  }
}
```

- **View generated MLIR code:**

add vscode settings to enable MLIR generated c++ code (suffix with .inc)syntax highlighting and code navigation.
```json
{
"C_Cpp.dimInactiveRegions": false
}
```

# Code structure overview

## MLIR triton dialect

### traits

How the Triton dialect defines and organizes its custom traits:

1. **TableGen definitions (`.td`)**  
   Custom Triton traits are defined in  
   `include/triton/Dialect/Triton/IR/TritonInterfaces.td`  
   as subclasses of `NativeOpTrait` and `NativeTrait` are MLIR meta-constructs that wrap C++ traits so they can be attached to ops via TableGen.
    All op traits ultimately derive from `Trait` in `mlir/IR/Traits.td`, which corresponds to `TraitBase` in the C++ side (`mlir/IR/OpDefinition.h`).

2. **C++ trait implementations**  
   Declared in `include/triton/Dialect/Triton/IR/Traits.h`, where each Triton trait is a subclass of `TraitBase` following this standard MLIR pattern:
   ```cpp
   template <typename ConcreteType>
   class TraitsName : public TraitBase<ConcreteType, TraitsName> {
   public:
     static LogicalResult verifyTrait(Operation *op) {
       return impl::verifyHelperFunc(op, ...);
     }
   };
    ```
3.	**Verification logic**
The actual verification functions in impl namespace are implemented in
`lib/Dialect/Triton/IR/Traits.cpp`.
The trait’s verifyTrait forwards to these helper functions.