# cuda_emu

A small fake CUDA runtime so the kernels can be tested on a machine without a GPU (like the
GitHub Actions runners). It only checks logic (indexing, edge cases, synchronization, numerics).
Performance numbers only make sense from `bench-kernels` on a real GPU.

How it maps things:

- each CUDA thread is an OS thread, and the blocks of a grid run one after another
- `__syncthreads()` is a `std::barrier` over the block
- warp shuffles go through a per-warp buffer plus a barrier. All 32 lanes have to take part, same as the full-mask `*_sync` intrinsics, so a divergent shuffle deadlocks here instead of quietly working
- `__shared__` turns into a function-local `static`. Only one block runs at a time, so that behaves like shared memory.
- `kernel<<<grid, block, smem, stream>>>(args)` and `extern __shared__` get rewritten by `preprocess.py`, and nothing else in the kernel source changes
- WMMA: every lane keeps the whole 16x16 tile, and `mma_sync` is computed once per warp and shared. All fragment types end up with the same element mapping, which is what the flash attention rescale trick relies on.
- CUDA graphs: launches made during stream capture are recorded and replayed

Run it with `make test-emu`. It takes a couple of minutes because of all the threads.
