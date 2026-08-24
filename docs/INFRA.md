# Optional: running the eval on a cloud GPU (GCP)

You don't need this to use vlm-eval — everything runs locally against Ollama or transformers. This is a
recipe for when you want production-grade GPU numbers (latency/VRAM on the card you'd actually deploy on).
Replace the ALL-CAPS placeholders with your own project/zone. Nothing here runs automatically.

Cost guard: 1×L4 `g2-standard-8` ≈ $0.85–0.97/h on-demand, ~60% of that as Spot (check the pricing
calculator on the day). Stop the VM when not running evals.

## 0. Pre-flight: does your project have GPU quota?

```bash
gcloud compute accelerator-types list --filter="name=nvidia-l4" --format="table(zone,name)"
gcloud compute regions describe YOUR_REGION --format="yaml(quotas)" | grep -B1 -A2 -E "NVIDIA_L4_GPUS|GPUS_ALL_REGIONS"
```

If the `NVIDIA_L4_GPUS` limit is `0.0`, request a quota bump (IAM & Admin → Quotas) first.

## 1. Create the VM (Spot, Deep Learning image with the NVIDIA driver pre-installed)

```bash
ZONE=YOUR_ZONE   # a zone from the accelerator-types list
gcloud compute instances create vlm-eval-l4 \
  --zone=$ZONE \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=250GB --boot-disk-type=pd-balanced \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT --instance-termination-action=STOP \
  --metadata=install-nvidia-driver=True
```

Then: `gcloud compute ssh vlm-eval-l4 --zone=$ZONE --tunnel-through-iap -- nvidia-smi`

## 2. Serve a model with vLLM (one at a time — the L4 has 24 GB)

```bash
docker run --rm --gpus all -p 8000:8000 -v /opt/hf:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-VL-8B-Instruct --served-model-name qwen3-vl-8b \
  --max-model-len 16384 --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image": 20}'
```

Cold start = time from `docker run` to `curl localhost:8000/health` returning 200 (second run, after the
weights are cached). VRAM: `nvidia-smi --query-gpu=memory.used --format=csv -l 5` in a second session.

## 3. Tunnel from your laptop; the harness talks to localhost

```bash
gcloud compute ssh vlm-eval-l4 --zone=$ZONE --tunnel-through-iap -- -N -L 8000:localhost:8000
# then: vlm-eval run --model qwen3-vl-8b --base-url http://localhost:8000/v1 --task tagging
```

vLLM (unlike Ollama) also gives you token logprobs — drop `--no-logprobs` to get real per-tag confidence.

## 4. Keep the weights somewhere (optional)

```bash
gsutil -m rsync -r /opt/hf/hub gs://YOUR_BUCKET/vlm-eval/hf-hub/
```

## 5. Stop when done

```bash
gcloud compute instances stop vlm-eval-l4 --zone=$ZONE
```
