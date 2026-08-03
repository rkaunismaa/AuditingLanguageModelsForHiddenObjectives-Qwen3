#!/usr/bin/env bash
# Serve a merged checkpoint via vLLM (OpenAI-compatible) in the serve env.
# Usage: scripts/serve_vllm.sh checkpoints/organism_final organism
#
# CUDA_DEVICE_ORDER=PCI_BUS_ID + CUDA_VISIBLE_DEVICES=1: this box mixes a
# GTX 1050 (physical/NVML index 0) with the RTX 4090 we actually train and
# serve on (physical index 1, per `nvidia-smi -L`). Without PCI_BUS_ID
# ordering, CUDA's default "fastest first" ordering and vLLM's own NVML-based
# quantization-capability check disagree on which device is "0" -- the
# capability check silently evaluates the 1050 and rejects quantization
# methods the 4090 actually supports. Pinning both env vars together makes
# every code path (torch/CUDA and vLLM's NVML checks) agree on device 1 =
# the 4090.
#
# QUANT/LOAD_FORMAT: Qwen3-14B is ~29GB in bf16, too large for the 4090's
# 24GB alone, so every checkpoint here needs 4-bit quantization to serve.
# QUANT defaults to bitsandbytes. Leave LOAD_FORMAT unset when serving an
# already-quantized checkpoint (e.g. unsloth/qwen3-14b-unsloth-bnb-4bit, our
# "base" stand-in); set LOAD_FORMAT=bitsandbytes when serving one of our own
# merged fp16 checkpoints (base_v1/base_v3/organism_final), which need to be
# quantized on the fly at load time.
#
# GPU_MEM_UTIL: vLLM's --gpu-memory-utilization preallocates this fraction of
# the GPU for weights + KV cache, sized by default (0.90) for high-concurrency
# serving. Our eval harness (run_eval.py) sends one generation at a time --
# no concurrency to provision for -- so 0.90 reserves far more KV cache than
# it ever uses (measured: 17x request concurrency at 0.90, ~24GB total).
# 0.55 was benchmarked as the practical floor: ~3.86x concurrency (still
# comfortable headroom above the 1x we need), ~15.5GB total, verified against
# a real generation. A lower value (0.48) also starts and generates
# correctly but leaves only ~1.22x concurrency -- too tight a margin for an
# unattended 1000-example eval run where a single response could approach
# max-model-len. Override for a different workload (e.g. concurrent serving).
set -euo pipefail
CKPT="${1:?checkpoint path}"; NAME="${2:-organism}"
QUANT="${QUANT:-bitsandbytes}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.55}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
.venv-serve/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name "$NAME" \
  --max-model-len 4096 --gpu-memory-utilization "$GPU_MEM_UTIL" --port 8000 \
  ${QUANT:+--quantization "$QUANT"} \
  ${LOAD_FORMAT:+--load-format "$LOAD_FORMAT"}
