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
- `tl.make_block_ptr` – create blocked pointer ensuring locality of BLOCK_SIZE_M x BLOCK_SIZE_N tiles
- `tl.dot(a, b, out_dtype=..., allow_tf32=...)` – matrix multiply intrinsic lowered through CPU pass pipeline

Demo script: `python/tutorials/cpu-blocked-matmul-aarch64.py`
- `block_transpose_combined_kernel`: transforms [M, N] -> [M/BM, N/BN, BM, BN] for page/locality friendly block access
- `matmul_kernel`: computes one output tile `[BLOCK_SIZE_M, BLOCK_SIZE_N]`


## packing implementation 

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

### python interface

our triton kernel packs a partial B matrix tile (tileK x tileN) into a packed format using the above C++ primitive interface.
```python
# example usage
rhs_packed = run_rhs_pack(BLOCK_SIZE_K, BLOCK_SIZE_N, rhs_stride, rhs_blk, bias_block, rhs_stride, rhs_blk, bias_block)

out: packed_rhs of shape (BLOCK_SIZE_K * BLOCK_SIZE_N + extra_bytes,)
```

## Triton CPU JIT Compiler (Python Perspective)

Focus: How Python code wraps, compiles, caches, and launches Triton kernels on CPU. (Lower-level MLIR/LLVM passes intentionally omitted.)

### Python-Level Flow (Source Function → Launched Kernel)
1. Decorate with `@triton.jit`: a `JITFunction` object captures the Python function, its signature, and constexpr annotations.
2. Indexing `fn[grid]` calls `JITFunction.__getitem__` returning a lightweight callable that stores the launch `grid`.
3. Calling that callable collects runtime args + keyword launch options (e.g. `num_threads`) and invokes `JITFunction.run`.
4. `run` builds a cache key (function source hash + constexpr values + CPUOptions hash) and looks up/creates a `CompiledKernel` via `compile()` in `python/triton/compiler/compiler.py`.
5. `compile()` drives a backend object (`CPUBackend`) which returns a `CompiledKernel` containing metadata plus a ready-to-use launcher handle supplied by `python/triton/backends/cpu/driver.py`.
6. The driver module’s generated C++ launcher (built on demand) executes an OpenMP loop over the flattened grid calling the native kernel entrypoint.

Result: Python invocation reuses cached native code, minimizing recompilation and providing a simple `kernel[grid](...)` API.

### Key Python Modules
- `python/triton/jit.py` (via `@triton.jit`): Wraps user function, implements `__getitem__`, caching logic, and argument marshalling.
- `python/triton/compiler/compiler.py`: Orchestrates compilation stages and caching (abstracts away IR specifics here).
- `python/triton/backends/cpu/compiler.py`: Backend interface exposed to Python; defines `CPUOptions` parsing and final artifact production.
- `python/triton/backends/cpu/driver.py`: Generates/compiles a small C++ launcher module; exposes a Python-callable `launch` function.

### CPUOptions (Runtime Launch Configuration)
Defined as a dataclass in `cpu/compiler.py` and parsed by `CPUBackend.parse_options`:
- `num_threads`: Requested max threads; 0 means use all available cores.
- `vec_lib`: Name of optional vector math library (string mapped to enum later).
- `ukernels`: Micro-kernel provider (e.g. OneDNN/XSMM); may be disabled if library not present.
- `enable_fast_math`: Influences math lowering decisions (abstracted here).
All fields contribute to the options hash used in the kernel cache key.

### Caching Mechanics (Simplified)
- Cache key components: function source digest + constexpr dict + options hash.
- On miss: build IR + native code, instantiate launcher, store `CompiledKernel`.
- On hit: reuse existing `CompiledKernel` without regenerating native artifacts.
- Environment knobs (Python-visible):
    - `TRITON_HOME`: Cache location base (`$TRITON_HOME/.triton`).
    - `MAX_JOBS`: Influences parallel build resource usage during initial native compilation.
    - `TRITON_CPU_UKERNELS_LIB`: Select micro-kernel provider (affects performance characteristics).

### Launch Semantics in Practice
```python
@triton.jit
def matmul_kernel(a, b, c, M, N, K,
                                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                                    OUT_DTYPE: tl.constexpr):
        # body omitted for brevity
        ...

grid = ( (M + BLOCK_M - 1)//BLOCK_M ) * ( (N + BLOCK_N - 1)//BLOCK_N )
matmul_kernel[grid](a, b, c, M, N, K, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, OUT_DTYPE=tl.float32, num_threads=8)
```
Steps performed:
1. `__getitem__` stores `grid`.
2. Call collects args; `num_threads=8` joins other kwargs.
3. Options parsed → `CPUOptions` (with `num_threads=8`).
4. Cache lookup → compile if needed → obtain launcher.
5. Launcher executes kernel body across grid indices using OpenMP with up to 8 threads.

### Driver (Runtime Glue) Overview
`driver.py` dynamically builds a tiny C++ module (includes runtime headers) that:
- Flattens the 3D grid to a linear range.
- Sets OpenMP thread count from metadata / `num_threads`.
- Performs per-program-id offset calculations before calling the compiled kernel function pointer.
Returned Python object exposes `launch(args...)` used transparently by the JIT wrapper.

### Extending From Python
To add new launch options or behaviors:
1. Extend `CPUOptions` dataclass (ensure new field participates in `.hash()`).
2. Accept the kwarg in your kernel call; `parse_options` will capture it.
3. If it affects runtime only (not codegen), plumb through launcher metadata and adjust C++ template in `driver.py`.

### Why This Abstraction Matters
- Keeps user API simple: grid + args + optional tuning knobs.
- Separates tuning (threads, micro-kernel choice) from algorithmic Python code.
- Enables rapid iteration: small Python changes only recompile when cache key changes.
- Avoids exposing internal IR/pass complexity to end users focusing on Python.

# triton language 

Triton is a DSL embedded in Python, but it uses syntactic interception, not runtime polymorphism. @triton.jit intercepts the code and converted it to AST for further compilation.

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
Provide the absolute path to `compile_commands.json` in C/C++ configuration for proper symbol navigation.
```bash
find python/build -name 'compile_commands.json' | xargs readlink -f
# example:
# python/build/cmake.linux-aarch64-cpython-3.12/compile_commands.json
```