#!/bin/bash
# Run a script from rs/ inside the vLLM image, offline and without network access.
# Usage: scripts/run_in_vllm.sh rs/agent_gen.py [args...]
# The model must already be in the Hugging Face cache (HF_HOME, default ~/.cache/huggingface).
set -e
IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:latest-cu130-ubuntu2404}
HF=${HF_HOME:-$HOME/.cache/huggingface}
exec docker run --rm --gpus all --ipc=host --network none \
  -v "$PWD":/work -w /work -v "$HF":/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/work/rs -e OMP_NUM_THREADS=4 \
  --entrypoint python3 "$IMAGE" "$@"
