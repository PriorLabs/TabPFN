# GPU Provisioning Steps (quick guide)

This document lists practical steps to provision GPU instances for TabPFN fine‑tuning. Use the pilot script to measure runtime before committing to long reservations.

## 1) Choose provider (quick guide)
- Low-cost / marketplace: Vast.ai, RunPod — good for short pilots and spot pricing.  
- Mid/Enterprise: CoreWeave, Lambda Labs — better SLAs, reserved capacity options.  
- Public cloud: AWS, GCP, Azure — use spot/preemptible for savings; reserved instances for production.

## 2) Quick pilot on RunPod or Vast.ai
1. Sign up and verify your account.  
2. Create a new instance (pod) and pick a GPU (e.g., A100, RTX 4090, H100 if available).  
3. Choose region close to you to reduce latency and egress costs.  
4. Add SSH key or use the web terminal.

Example install + run commands (once instance is ready):
```bash
# update + create python venv
sudo apt-get update && sudo apt-get install -y python3-venv git
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install tabpfn torch pandas scikit-learn tqdm
# copy repo or upload required files
git clone <your-repo-url> repo && cd repo
python scripts/pilot_timing.py --n_pilot 1000
```

## 3) Using AWS/GCP spot instances (fast example)
- AWS (spot, single‑GPU quick start):
  1. Create keypair and security group with SSH access.  
  2. In EC2 console, `Launch Instance` → choose Deep Learning AMI or Ubuntu + CUDA.  
  3. Select instance type (e.g., `p4d` for A100 clusters; `g5`, `g5g`, `g5a` for other NVIDIA GPUs).  
  4. Under `Purchasing option`, choose `Request Spot instance` and set bid behaviour.  
  5. SSH in and run the same install/run commands as above.

- GCP (preemptible): use `gcloud` or console to create an accelerator‑attached VM and enable preemptible flag.

## 4) Checkpointing & cost control
- Frequent checkpoints to object storage (S3, GCS) to allow resuming after spot/preemptible eviction.  
- Monitor usage and enable alerts / spend limits in the provider console.  
- Use smaller test runs (1–3 epochs) for debugging, then scale.

## 5) Storage & egress
- Store large checkpoints on object storage (S3/GCS/CoreWeave object store).  
- Minimize egress by performing final aggregation on the instance or copying only small artifacts (metrics, CSVs).  

## 6) Production runs
- For final reproducible runs, prefer on‑demand or reserved instances to avoid interruptions.  
- If you expect regular usage, request a reserved capacity quote from CoreWeave or your cloud provider for discounts.

## 7) Common pitfalls
- Insufficient VRAM: reduce batch/context sizes or pick a larger GPU.  
- No checkpointing: spot/preemptible interruptions will require re‑runs.  
- Missing GPU drivers/CUDA mismatch: use provider Deep Learning AMIs or driver‑matched images.

---
If you want, I can produce a step‑by‑step RunPod or Vast.ai runbook tied to your account (I’d need the chosen provider and preferred GPU type).

## 8) Provider-specific runbook & quick commands
Below are reproducible steps and example commands you can copy/paste for common providers. Adjust instance types and pricing to your account/region.

RunPod / Vast.ai (fast pilot, spot-style)
- Pick GPU: `RTX-4090`, `A100-40GB`, or `A100-80GB` (if available).
- Recommended: start with a 1‑hour spot/pod reservation to run the pilot and debug.

Example (after instance is ready & you have a shell):
```bash
# create venv and install deps
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install tabpfn torch pandas scikit-learn tqdm

# (optional) set HF token if model is gated (create at https://huggingface.co/settings/tokens)
export HF_TOKEN=hf_...

# clone your repo or upload scripts
git clone <your-repo-url> repo && cd repo

# Run the lightweight timing script (GPU)
python scripts/pilot_timing.py --n_pilot 2000 --device cuda

# Or run the finetune pilot (single epoch) for a larger end-to-end check
python scripts/finetune_pilot.py --n_pilot 10000 --device cuda --epochs 1
```

CoreWeave / Lambda (enterprise, reliable)
- Request an `A100-40GB` or `A100-80GB` instance for predictable performance. Use their web console to reserve 1–2 hours for pilot.
- Use the same in-instance commands above to run the pilot scripts.

AWS (on‑demand / spot) quick notes
- Use Deep Learning AMIs (drivers + CUDA packaged). Instance families: `p4`, `p4d`, or `g5`/`g5dn` depending on availability.
- Request a spot instance for cost savings; add a 20–30% contingency for preemptions.

Hugging Face gated models & model acceptance
- If TabPFN weights are gated, visit the model page (e.g. https://huggingface.co/Prior-Labs/tabpfn_2_6) and accept terms while logged in.
- Then either run `huggingface-cli login` or set `HF_TOKEN` in the shell before running the scripts.

Cost & timing quick-reference (use measured CPU baseline to refine)
- Measured CPU baseline: time_per_sample ≈ 0.01875 s → single full fine-tune (N=10k, epochs=10) ≈ 0.52 CPU-hr.
- Use pilot run on your chosen GPU to measure `time_per_sample_gpu` and compute:
  GPU_hours = time_per_sample_gpu * N * epochs / 3600

Sample spot rate snapshot (approximate; confirm current prices):
- RTX4090 (spot): ~$0.25–0.50 / hr
- A100-40GB (spot): ~$0.7–1.2 / hr
- A100-80GB (spot): ~$1.2–1.8 / hr
- H100 (spot): ~$2.0–4.0 / hr

Budgeting tip: for a conservative procurement request, ask for 5–10 GPU‑hours (covers pilot + small HPO), and request spot pricing with a 20–30% contingency.

## 9) Checkpointing / resilience (recommended)
- Save model checkpoints and important artifacts to object storage (S3/GCS/CoreWeave bucket) frequently (every epoch or every N minutes).
- Example copy to S3:
```bash
# configure aws cli or use provider SDK
aws s3 cp /path/to/checkpoint.ckpt s3://my-bucket/tabpfn-checkpoints/ --acl private
```

## 10) Quick troubleshooting
- If you see model download errors: ensure `HF_TOKEN` is set and you accepted the model terms.
- If CUDA errors appear: check driver + CUDA toolkit versions or use the provider's Deep Learning AMI.

---
If you want, I will now produce a one‑page procurement-ready budget (option: A100-80 spot, 10 GPU-hours, 30% contingency) or prepare a provider-specific runbook for RunPod or CoreWeave. Tell me which and I’ll generate the document.