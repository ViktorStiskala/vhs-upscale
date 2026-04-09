# AI-powered VHS restoration in 2025: models, pipeline, and cloud setup

**The optimal VHS restoration pipeline chains a VHS-specific 2x model with a general-purpose 2x upscaler for 4x total, replaces DPIR with SCUNet for denoising, and adds a dedicated chroma cleanup pass — all running on an A100 SXM 80GB ($1.39/hr on Runpod) with CUDA 12.4.1.** No single 4x model trained specifically on VHS content exists as of early 2026, making the two-stage chain approach not just a workaround but the technically superior strategy. The VapourSynth ecosystem has matured significantly, with spandrel supporting 40+ architectures and new GPU-accelerated deinterlacers emerging as QTGMC alternatives. This report covers every component of a state-of-the-art VHS restoration pipeline for your exact hardware constraints.

---

## VHS-specific restoration models have quietly proliferated

The most significant purpose-built model is **TAPE** (Reference-based Restoration of Digitized Analog Videotapes, WACV 2024), a Swin-UNet architecture with multi-reference spatial feature fusion that uses CLIP-based zero-shot artifact detection to find clean reference frames. TAPE specifically targets tape mistracking, VHS edge waving, chroma loss, tape noise, and undersaturation. It is available at github.com/miccunifi/TAPE with code, models, and synthetic training data. However, TAPE is a standalone PyTorch pipeline with no VapourSynth integration — using it requires either a custom wrapper or a frame-export workflow.

For models that slot directly into your existing vsspandrel pipeline, OpenModelDB offers several VHS-trained options:

- **1x BleedOut Compact** (Zarxrax, SRVGGNetCompact 64nf16nc): Specifically repairs color bleed, heavy chroma noise, and VHS rainbows. At 4.6 MB, it runs nearly instantaneously and operates at 1x scale (restoration only, no upscaling). This is the single most impactful addition to your current pipeline — a targeted chroma artifact remover that runs via vsspandrel with zero friction.
- **2xVHS2HD-RealPLKSR** (asterixcool): Your current model. Trained on 6,019 PAL VHS pairs across 385K iterations. Handles haloing, ghosting, and noise while upscaling 2x. Remains the best VHS-specific upscaling model available.
- **2x VHS Upscale and Denoise Film** (Itewreed, ESRGAN): Trained specifically on VHS captures of film material paired with Blu-ray rips. Worth testing if your source is recorded movies rather than native camcorder footage.

For composite video artifacts (dot crawl, rainbow), the **NES/Genesis Composite-2-RGB** OmniSR models from pokepress explicitly note that the OmniSR variant handles VHS and RF content better than the compact variant, though these are trained primarily on game console output.

### SCUNet should replace DPIR as your denoiser

Your current DPIR/DRUNet denoiser was trained primarily on additive white Gaussian noise, which is a poor match for VHS's complex noise profile of analog interference patterns, chroma blotching, and signal-dependent noise. **SCUNet** (Swin-Conv-UNet) was trained with a realistic complex noise model including JPEG artifacts, sensor noise, and mixed degradation types. Doom9 forum users processing S-VHS captures report "deep denoise but not too much loss of details" with the `scunet_color_real_psnr` model variant.

SCUNet has first-class VapourSynth support through three paths: HolyWu's `vs-scunet` (`pip install -U vsscunet`), AmusementClub's `vs-mlrt` with built-in SCUNet support and TensorRT acceleration, and direct loading via vsspandrel. The `scunet_color_real_psnr` variant is recommended over `scunet_color_real_gan` for VHS — the GAN variant introduces color shifts that compound with VHS's already-inaccurate color.

**NAFNet** achieves the highest benchmark PSNR (40.30 dB on SIDD vs DPIR's 39.85 dB) at under 50% of DPIR's computational cost, but it was trained on smartphone sensor noise (SIDD dataset) and would need fine-tuning for VHS. **Restormer** is supported via `spandrel_extra_arches` but is computationally expensive with no VHS-specific advantage over SCUNet. For practical VHS denoising in your pipeline, SCUNet is the clear upgrade — better noise model, equivalent VapourSynth integration, and similar speed to DPIR.

### Temporal denoisers offer the biggest quality leap

Single-frame denoisers like DPIR and SCUNet discard the most valuable signal available in video: temporal redundancy across frames. For VHS content where noise varies frame-to-frame but the underlying signal persists, multi-frame approaches yield dramatically better results.

**BM3DCUDA** (VapourSynth-BM3DCUDA by WolframRhodium) is the most practical temporal denoiser for your pipeline — GPU-accelerated BM3D with configurable temporal radius, widely used in the VHS restoration community, and requiring no model conversion. The recommended hybrid approach: BM3DCUDA for temporal+spatial denoising on luma, KNLMeansCL for aggressive chroma denoising.

**vs-basicvsrpp** (HolyWu, updated January 2025) runs BasicVSR++ in VapourSynth with models 3-5 providing same-size restoration (no upscaling). However, community notes indicate it works best on "fairly clean sources" — heavy VHS noise may need pre-filtering. **vs-rvrt** exists (by Lyra-Vhess) for RVRT denoising with sigma 0-50, FP16, and tiling support, though it's VRAM-hungry and trained on Gaussian noise.

The most innovative approach is **TAP** (Temporal As a Plugin, ECCV 2024), which augments any pre-trained image denoiser with temporal modules at skip connections without retraining the base model. This could theoretically lift your SCUNet from single-frame to multi-frame denoising, but it remains research code with no VapourSynth integration.

---

## No 4x VHS model exists — chain two 2x passes instead

A thorough search of OpenModelDB, GitHub, and community repositories confirms that **no 4x model trained specifically on VHS content exists**. The community has produced 2x VHS models but stopped there, likely because VHS's low information density makes 4x a more speculative reconstruction task better handled in stages.

### The two-stage chain is technically superior

The recommended approach for 720×576 → 2880×2304 is:

1. **First pass**: 2xVHS2HD-RealPLKSR (720×576 → 1440×1152) — handles VHS-specific artifacts
2. **Second pass**: A general-purpose 2x upscaler (1440×1152 → 2880×2304) — cleanly doubles already-restored content

This outperforms a single 4x pass because the first model operates in the VHS degradation domain it was trained on, while the second model receives already-cleaned input that matches its general-purpose training distribution. Multiple Topaz community users confirm that two-pass processing with different objectives per pass ("first pass for compression reversion and deblur, second pass for noise reduction") yields better quality than single-pass.

For the second 2x pass, the best options ranked by quality-to-speed ratio:

- **2x SPAN models** (e.g., 2xNomosUni_span_multijpg): **11x faster than ESRGAN**, good quality. Best for bulk processing. ~20-50 fps on A100 at 1440×1152 input.
- **2x RealPLKSR models** (e.g., 2xNomosUni_compact_multijpg): Good balance. ~10-20 fps on A100.
- **2x Compact/SRVGGNet** models: Fastest architecture. ~25+ fps on A100. Slightly lower quality ceiling.

For users wanting a single 4x pass after separate restoration, the best general-purpose 4x models are **4xNomos2_realplksr_dysample** (best quality/speed in the RealPLKSR family, uses DySample for cleaner upsampling), **4xUltraSharpV2** (DAT2 architecture, Kim2091's flagship model trained for years on diverse data — highest quality but ~1-4 fps on A100), and **4xNomosUni_span_multijpg** (fastest 4x architecture at ~20-50 fps).

### VRAM is not a constraint on A100 80GB

At 720×576 input resolution, even the heaviest architectures fit comfortably in 80GB VRAM without tiling. RealPLKSR uses **3-5 GB in FP16**, ESRGAN uses **4-8 GB**, and even HAT-L (the largest common architecture at ~40M parameters) uses only **14-24 GB**. The chained approach processes 720×576 then 1440×1152 — both well within limits. TensorRT conversion via vs-mlrt further reduces VRAM by 30-50% while roughly doubling throughput.

---

## VapourSynth's AI ecosystem now covers nearly everything

**Spandrel** (chaiNNer-org/spandrel) auto-detects and loads 40+ architectures from `.pth` files. Every single-image model discussed in this report is supported: RealPLKSR, ESRGAN/RealESRGAN, SwinIR, HAT, DAT, SPAN, OmniSR, ATD, DRUNet, NAFNet, SCUNet, and Restormer (via `spandrel_extra_arches`). The critical limitation: **spandrel only supports single-image models**. Temporal/video models like BasicVSR++, RVRT, and VRT require dedicated wrappers.

### Plugin compatibility matrix (what works today)

| Model category | Via vsspandrel | Via dedicated plugin | Via vs-mlrt (TensorRT) |
|---|---|---|---|
| RealPLKSR, ESRGAN, SwinIR, HAT, DAT, SPAN, OmniSR, ATD | ✅ All supported | — | ✅ Via ONNX conversion |
| DRUNet (DPIR) | ✅ | ✅ vs-dpir (HolyWu) | ✅ Built-in |
| SCUNet | ✅ | ✅ vs-scunet (HolyWu) | ✅ Built-in |
| NAFNet, Restormer | ✅ | — | Via ONNX conversion |
| BasicVSR++ | ❌ Temporal | ✅ vs-basicvsrpp (HolyWu) | — |
| RVRT | ❌ Temporal | Partial (vs-rvrt by Lyra-Vhess) | — |
| RIFE (interpolation) | N/A | ✅ vs-rife (HolyWu, updated Feb 2026) | ✅ Built-in |
| MFDIN (AI deinterlace) | N/A | ✅ vs-mfdin (HolyWu, Dec 2024) | — |

**AmusementClub/vs-mlrt** is the premier inference runtime, now at v15.x with TensorRT 10.14.1, CUDA 13.0.2, cuDNN 9.13.0. It supports TensorRT (fastest on NVIDIA), ONNX Runtime, OpenVINO, and ncnn backends. Any ONNX model can be loaded via its generic `vsmlrt.inference()` API, making it the fastest path for running OpenModelDB models on A100/H100.

### GPU-accelerated deinterlacing alternatives to QTGMC

Two new options have emerged for GPU deinterlacing in VapourSynth:

- **vs-mfdin** (HolyWu, December 2024): Based on MFDIN (Multiframe Joint Enhancement for Early Interlaced Videos), which led the videoprocessing.ai deinterlacer benchmark. Supports TensorRT via torch_tensorrt. The most promising QTGMC replacement for quality.
- **vs_deepdeinterlace** (pifroggi): Three ML deinterlacers (DDD, DeF, DfConvEkSA) running on CUDA. DDD runs at ~50fps on RTX 4090 at 480p. Can be used as `EdiExt` clip in QTGMC for a hybrid approach — using ML edge interpolation within QTGMC's motion compensation framework.

Neither fully replaces QTGMC for archival VHS work yet, but vs-mfdin with TensorRT on an A100 could match or exceed QTGMC quality while shifting the load to GPU.

---

## The optimal pipeline separates chroma and luma processing

VHS stores luma at ~3.0 MHz bandwidth and chroma at only **~0.6 MHz** — chroma is roughly 5x lower resolution. This fundamental asymmetry means aggressive chroma denoising preserves almost nothing real, while even moderate luma denoising risks destroying genuine detail. Every expert source recommends separate processing.

### Recommended pipeline order with specific filters

The consensus order, refined from VideoHelp, doom9, and vhs-decode community expertise:

1. **Chroma shift correction** (before deinterlacing — field-level alignment matters): Use `fmtc.resample` with `sx` parameter for sub-pixel chroma shifts, typically -2 to -6 pixels for PAL VHS.

2. **Chroma bleed fix** (before deinterlacing): Run **1x BleedOut Compact** via vsspandrel for AI-based color bleed removal, or use aWarpSharp2 on chroma only (`core.std.MergeChroma(clip, core.warp.AWarpSharp2(clip, depth=20, chroma=6))`). Experts on VideoHelp prefer the aWarpSharp2 approach over FixChromaBleedingMod.

3. **Deinterlace** (QTGMC, "Slow" or "Slower" preset): Enable `ChromaNoise=True`, `ChromaMotion=True`, `DenoiseMC=True`, `EZDenoise=3.0`. QTGMC inherently performs temporal denoising as a byproduct of motion compensation.

4. **Chroma denoise** (aggressive, GPU): `core.knlm.KNLMeansCL(clip, a=2, h=3.0, d=3, channels='UV', device_type='gpu')` — apply only to UV planes with high strength.

5. **Luma denoise** (moderate): SCUNet via vs-scunet (`scunet_color_real_psnr` model) or BM3DCUDA with temporal radius 1-2. Process before upscaling to prevent noise amplification.

6. **Upscale pass 1**: 2xVHS2HD-RealPLKSR via vsspandrel (720×576 → 1440×1152).

7. **Upscale pass 2**: 2x SPAN or RealPLKSR general model via vsspandrel (1440×1152 → 2880×2304).

8. **Post-upscale sharpen**: CAS at 0.3-0.5 on luma only (`core.cas.CAS(clip, sharpness=0.4, planes=[0])`). AI upscalers sometimes produce slightly soft output; CAS restores perceived sharpness without harsh oversharpening.

9. **Optional grain restoration**: `core.grain.AddGrain(clip, var=0.5, uvar=0.0)` to prevent the "plastic" look common after heavy denoising. QTGMC's `GrainRestore` parameter can also handle this at step 3.

10. **Color space conversion**: PAL VHS uses BT.601 (470bg). Convert to BT.709 when outputting HD: `core.resize.Bicubic(clip, matrix_in_s="470bg", matrix_s="709")`.

For the denoise-before-vs-after-upscale debate: the majority expert view is to denoise before upscaling, but an alternative approach from DTL2023 on VideoHelp (May 2024) suggests that with AI upscalers, denoising after upscale can preserve more fine detail because "any small amount of fine details become much larger relative to sample size." For VHS with heavy noise, the practical answer is a moderate denoise before upscaling plus an optional light cleanup after.

---

## Runpod setup: A100 SXM at $1.39/hr with CUDA 12.4.1

### GPU selection and pricing

| GPU | VRAM | vCPUs | On-demand/hr | Best for |
|---|---|---|---|---|
| A100 PCIe 80GB | 80 GB | 8 | $1.19 | Budget option (fewer vCPUs limit QTGMC) |
| **A100 SXM 80GB** | **80 GB** | **16** | **$1.39** | **Best value — 16 vCPUs for QTGMC + 80GB for AI models** |
| H100 PCIe 80GB | 80 GB | 16 | $1.99 | ~1.5-2x faster GPU inference, same vCPUs |
| H100 SXM 80GB | 80 GB | 20 | $2.69 | Maximum GPU speed + most vCPUs |

The **A100 SXM 80GB at $1.39/hr** offers the best cost-effectiveness. Its 16 vCPUs are critical — QTGMC is CPU-bound and scales well across threads, and the A100 PCIe's 8 vCPUs would bottleneck deinterlacing. The H100 provides ~1.5-2x faster AI inference through higher tensor core throughput and FP8 support, but at 43% higher cost — worthwhile only if GPU inference (upscaling, denoising) dominates your processing time rather than QTGMC.

Use **on-demand pods, not spot instances**. Spot instances give only a 5-second SIGTERM warning before termination — insufficient to save a multi-hour video encoding job. Network volumes at **$0.07/GB/month** (under 1TB) are persistent and portable between pods, with no data egress fees. Use network volumes for source videos and outputs; use local container disk for working files during processing.

### Docker and Broadwell compatibility

The recommended base image is:
```
nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
```

CUDA 12.4.1 has the broadest compatibility with PyTorch 2.4+, FlashAttention, TensorRT, and the VapourSynth ML ecosystem. Always use the `-devel` tag (not `-runtime`) since you need compilation headers for building VapourSynth plugins. Runpod Secure Cloud A100/H100 pods run NVIDIA driver 550+, which fully supports CUDA 12.4.

Regarding Broadwell CPUs: **standard PyTorch GPU builds work correctly** on Broadwell with AVX2. PyTorch's pre-built wheels include runtime CPU dispatch and automatically use AVX2 codepaths. No special build flags are needed. QTGMC's dependencies (FFTW3, MVTools2, NNEDI3) all use SSE2/AVX2 codepaths with no AVX-512 requirement. The one caveat: some packages compiled with `-march=native` on AVX-512 machines may crash with "Illegal instruction" on Broadwell — **always build inside the Docker container** or use pre-built wheels.

That said, Runpod's Secure Cloud A100/H100 pods typically use modern EPYC or Xeon Scalable CPUs, not Broadwell. Broadwell concerns are most relevant if you're preprocessing on a separate machine or using Community Cloud hosts with older hardware.

### QTGMC tuning for your vCPU budget

With 16 vCPUs on A100 SXM, set `core.num_threads = 16` in VapourSynth. The "Slow" preset is the best quality-speed tradeoff for VHS, yielding ~8-12 fps on SD content with 16 threads. The settings with the largest CPU impact are **TR2** (temporal radius, keep at 1), **SourceMatch** (set to 3 for archival, 0 for speed), and **NoisePreset** (match to your main preset). Setting `Sharpness=0.0` in QTGMC is advisable when you plan to apply CAS after upscaling — it avoids double-sharpening.

Estimated processing cost: approximately **$5-7 per hour of PAL VHS content** on A100 SXM, including QTGMC deinterlacing (~3 hours), AI denoising and dual upscaling (~1.5 hours), and FFmpeg encoding (~0.5 hours).

---

## Conclusion

The VHS restoration landscape in 2025-2026 has matured into a capable but fragmented ecosystem. Three changes will most improve your existing pipeline: replacing DPIR with **SCUNet** for real-world noise handling, adding **1x BleedOut Compact** before denoising for targeted chroma artifact removal, and chaining your existing 2xVHS2HD with a fast general-purpose **2x SPAN model** for 4x total upscaling. The absence of any 4x VHS-specific model makes the two-stage chain the technically correct approach, not a compromise.

The emerging frontier is GPU deinterlacing: **vs-mfdin** could eventually replace QTGMC and eliminate the CPU bottleneck entirely, shifting the entire pipeline to GPU. For now, the hybrid approach — QTGMC on CPU for proven deinterlacing quality, AI models on GPU for everything else — remains the most reliable path. The A100 SXM 80GB on Runpod provides the optimal balance of GPU VRAM, vCPU count for QTGMC threading, and cost-effectiveness at $1.39/hour. Every model discussed here runs through either vsspandrel or vs-mlrt with TensorRT acceleration, keeping the pipeline unified within VapourSynth.
