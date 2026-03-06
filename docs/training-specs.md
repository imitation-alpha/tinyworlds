# TinyWorlds Training Hardware Specifications

## Default Model Sizes

| Stage | Model | Blocks | Embed Dim | Hidden Dim | Heads | Est. Parameters | Training Steps |
|-------|-------|--------|-----------|------------|-------|-----------------|----------------|
| 1 | Video Tokenizer (FSQ-VAE) | 4 | 32 | 128 | 8 | ~500K–1M | 40,000 |
| 2 | Latent Actions (FSQ-VAE) | 2 | 32 | 128 | 8 | ~250K–500K | 10,000 |
| 3 | Dynamics (Transformer) | 8 | 32 | 128 | 8 | ~1M–2M | 300,000 |

Total across all stages: **~2–3.5M parameters** (very small by modern standards).

## Minimum Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM / VRAM | 4 GB | 8+ GB |
| Storage | 5 GB | 20 GB (for datasets) |
| GPU | Optional (CPU works) | Any CUDA or MPS GPU |

## Memory Footprint Estimate (Default Config)

- **Model weights** (FP32): < 10 MB per stage
- **Batch** (350 frames × 64×64 × 3ch × FP32): ~170 MB
- **Optimizer state** (Adam, 2 momentum buffers): ~3× model size ≈ 30 MB
- **Activations + gradients**: ~200–500 MB
- **Peak memory per stage**: **< 2 GB**

## Apple Silicon (MPS) Compatibility

TinyWorlds can run on Apple Silicon Macs using PyTorch's MPS backend. Required config changes in `configs/training.yaml`:

```yaml
# Disable NVIDIA-specific features
amp: false          # or test — MPS AMP support is improving
tf32: false         # TF32 is NVIDIA Tensor Core specific
compile: false      # torch.compile has limited MPS support

# Keep these off (multi-CUDA-GPU features)
distributed:
  use_ddp: false
  use_fsdp: false
```

The training scripts use CUDA device detection in `utils/distributed.py`. For MPS, ensure your device selection falls back to `mps` when CUDA is unavailable. PyTorch 2.8+ handles this with:

```python
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
```

### Mac Mini M4 Pro 64 GB — Assessment

**Verdict: More than capable.**

- 64 GB unified memory is shared between CPU and GPU — roughly equivalent to having 64 GB of VRAM
- Default models need < 2 GB peak memory
- You could scale models 10–50× larger and still fit comfortably

**Estimated training times on M4 Pro (MPS):**

| Stage | Steps | Est. Time |
|-------|-------|-----------|
| Video Tokenizer | 40K | 10–30 min |
| Latent Actions | 10K | 5–15 min |
| Dynamics | 300K | 2–6 hours |

### Scaling on 64 GB Unified Memory

With 64 GB, you can push well beyond defaults:

| Parameter | Default | Safe Max (64 GB) |
|-----------|---------|-------------------|
| `embed_dim` | 32 | 256–512 |
| `hidden_dim` | 128 | 1024–2048 |
| `num_blocks` | 4–8 | 16–32 |
| `frame_size` | 64 | 128–256 |
| `batch_size_per_gpu` | 350–500 | 64–128 (with larger models) |

This would produce models in the **10M–100M parameter** range.

## NVIDIA GPU Training

For CUDA GPUs, all performance features work out of the box:

- **AMP** (`amp: true`): Mixed precision with BF16 — halves memory, faster math
- **TF32** (`tf32: true`): TensorFloat32 on Ampere+ tensor cores
- **torch.compile** (`compile: true`): Fused CUDA kernels
- **DDP/FSDP**: Multi-GPU distributed training

| GPU | VRAM | Can Train Default? | Can Scale? |
|-----|------|--------------------|------------|
| GTX 1060 6GB | 6 GB | Yes | Moderate |
| RTX 3060 12GB | 12 GB | Yes | Good |
| RTX 4090 24GB | 24 GB | Yes | Excellent |
| A100 40/80GB | 40–80 GB | Yes | Maximum |

## CPU-Only Training

Training on CPU is supported but significantly slower. Useful for debugging or very small experiments. Disable all GPU-specific optimizations:

```yaml
amp: false
tf32: false
compile: false
```

Reduce batch sizes to 16–64 for reasonable training times.
