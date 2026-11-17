# Serving a model with vLLM on a cloud GPU

This is optional. Everything in vlm-eval runs locally against Ollama or an in-process checkpoint. Use
this recipe when you want latency, throughput and memory figures from the kind of GPU you would deploy
on, or when you want token logprobs, which vLLM exposes and Ollama does not.

**Status.** The harness side of this path (the `vllm` flavor, schema-constrained output, logprobs) is
covered by mocked tests and has not been run against a live vLLM server. The cloud commands below are
an example for Google Cloud with one NVIDIA L4; they are not executed or checked by anything in this
repository. Machine types, image families, vLLM flags and prices change, so check each against the
provider's and vLLM's current documentation. Nothing here runs automatically. Replace the upper-case
placeholders with your own values.

A GPU VM is billed while it runs. Check the current price before starting and stop the VM when you are
done.

## 0. Check GPU quota

```bash
gcloud compute accelerator-types list --filter="name=nvidia-l4" --format="table(zone,name)"
gcloud compute regions describe YOUR_REGION --format="yaml(quotas)" | grep -B1 -A2 -E "NVIDIA_L4_GPUS|GPUS_ALL_REGIONS"
```

If the L4 limit is zero, request a quota increase first.

## 1. Create the VM

Use an image with the NVIDIA driver preinstalled. List the current Deep Learning VM image families
with `gcloud compute images list --project deeplearning-platform-release` and pick one.

```bash
ZONE=YOUR_ZONE
gcloud compute instances create vlm-eval-l4 \
  --zone=$ZONE \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=YOUR_IMAGE_FAMILY \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=250GB --boot-disk-type=pd-balanced \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT --instance-termination-action=STOP \
  --metadata=install-nvidia-driver=True

gcloud compute ssh vlm-eval-l4 --zone=$ZONE --tunnel-through-iap -- nvidia-smi
```

## 2. Serve one model

On the VM. The server stays in the foreground.

```bash
docker run --rm --gpus all -p 8000:8000 -v /opt/hf:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-VL-8B-Instruct --served-model-name qwen3-vl-8b \
  --max-model-len 16384 --gpu-memory-utilization 0.92 \
  --limit-mm-per-prompt '{"image": 20}'
```

The image limit of 20 matches the most images the summary task sends in one call.

Two figures are worth recording by hand for the model card. Cold start is the time from `docker run`
until `curl localhost:8000/health` returns 200, on a second start when the weights are already cached.
Peak memory is what `nvidia-smi --query-gpu=memory.used --format=csv -l 5` shows during a run.

## 3. Tunnel and run the harness from your machine

```bash
gcloud compute ssh vlm-eval-l4 --zone=$ZONE --tunnel-through-iap -- -N -L 8000:localhost:8000
```

Then, in another terminal:

```bash
vlm-eval run qwen3-vl-8b-vllm tagging --limit 1 \
    --served-name qwen3-vl-8b --base-url http://localhost:8000/v1 --flavor vllm
```

Or add a preset to `models.local.json` (see `models.local.example.json`) and use its name. Logprobs
are requested by default; `--no-logprobs` turns them off. With them on, each tagging row has a
`confidence` map with the probability of `true` per tag.

A vLLM endpoint reports a model name but no weights digest, so the harness lets a fresh run proceed and
refuses to resume a run file that already has rows ([design.md](design.md#provenance)). Plan runs on
this path to complete in one go, or archive the partial file before restarting.

For throughput on this hardware:

```bash
vlm-eval perf qwen3-vl-8b-vllm --served-name qwen3-vl-8b --base-url http://localhost:8000/v1 \
    --flavor vllm --concurrency 4 --n 40
```

## 4. Keep the weights (optional)

```bash
gsutil -m rsync -r /opt/hf/hub gs://YOUR_BUCKET/vlm-eval/hf-hub/
```

## 5. Stop the VM

```bash
gcloud compute instances stop vlm-eval-l4 --zone=$ZONE
```
