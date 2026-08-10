# Environment notes (tested combination)

All results were produced with the following stack; nearby versions likely
work but have not been tested.

| Component | Version |
|---|---|
| GPU | NVIDIA V100-SXM2-32GB (1 GPU per run) |
| CUDA toolkit / driver runtime | 12.1 |
| Python | 3.10 |
| PyTorch | 2.4.0+cu121 |
| gsplat | 1.5.3 |
| numpy / scipy / scikit-image | 2.2.6 / 1.15.3 / 0.25.2 |
| Pillow / OpenCV | 12.2.0 / 5.0.0 |
| lpips | 0.1.4 |
| transformers (Depth Anything V2) | 4.45.0 |

Practical notes:

- **gsplat JIT compilation**: gsplat compiles its CUDA kernels on first use;
  it needs a host compiler compatible with your CUDA toolkit (we used a
  GCC 10/11 toolchain with `CC`/`CXX`/`CUDA_HOME` exported; the OS-default
  GCC 7 on older distributions fails to build the kernels).
- **Depth Anything V2** weights (`depth-anything/Depth-Anything-V2-Small-hf`)
  are downloaded from the Hugging Face Hub on first run of
  `tools/gen_radar_anchored_depth.py` (~100 MB, cached afterwards).
- **LPIPS** downloads AlexNet weights on first evaluation run.
- Memory: 30k-iteration training on 1280x720 frames peaks well below 32 GB
  GPU memory with the benchmark config; 16 GB cards should also work.
- No network access is needed after the two weight downloads above.
