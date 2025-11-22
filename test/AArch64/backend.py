try:
    import torch
    import triton
    import triton.language.semantic as semantic
    import triton.language as tl
    from triton import jit
except Exception as e:
    print("Failed to import local `triton` package. Ensure you run from the repository root and that `python/` is present.")
    raise

DEVICE = "cpu"
device = torch.device(DEVICE)

@jit
def store_kernel(x, M: tl.constexpr):
    # Make a simple tensor descriptor and load a block (this exercises descriptor/block load lowering)
    tl.store(x, 1)
    val = tl.load(x)
    tl.abs(val)


def test_store_kernel():
    M = 3
    x = torch.zeros((M), device=device, dtype=torch.float32)
    grid = (1,)
    # pass MB and NB as keyword constexpr arguments
    store_kernel[grid](x, M)
    print("After store_kernel:")
    print(x)


def test_libdevice_abs():
    import importlib
    cpu_lib = importlib.import_module("triton.language.extra.cpu.libdevice")
    top_lib = importlib.import_module("triton.language.extra.libdevice")
    print("tl.abs:", tl.abs, getattr(tl.abs, "__module__", None))
    print("top libdevice.abs:", top_lib.abs)
    print("cpu libdevice.abs:", cpu_lib.abs)
    print("is extern?", getattr(cpu_lib.abs, "__triton_extern__", getattr(cpu_lib.abs, "__triton_builtin__", None)))

def test_run_rhs_pack():
    @jit
    def rhs_pack_kernel(rhs, bias, M: tl.constexpr, N: tl.constexpr):
        packed_rhs = tl.pack_rhs(rhs, bias)
        # tl.store(packed_rhs, 1)

    M = N = 32
    x = torch.zeros((M, N), device=device, dtype=torch.float32)
    bias = torch.zeros((N), device=device, dtype=torch.float32)
    grid = (1,)
    rhs_pack_kernel[grid](x, bias, M, N)
    print("After rhs_pack_kernel:")
    print(x)

if __name__ == "__main__":
    test_run_rhs_pack()
    # test_store_kernel()
    # test_libdevice_abs()