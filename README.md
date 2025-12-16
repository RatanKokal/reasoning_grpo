# EfficientMath-GRPO

Verifier-guided GRPO for improving mathematical reasoning accuracy while controlling generation length and serving latency.

The project compares three reward designs after LoRA supervised fine-tuning:

1. Correctness-only GRPO
2. Fixed-length GRPO
3. Hardware-calibrated adaptive-latency GRPO

The default model is `Qwen/Qwen2.5-0.5B-Instruct`, trained in BF16 on a single RTX 6000-class GPU.

## Results

Evaluated on the full 1,319-example GSM8K test set:

| Model                 |  Accuracy | Mean output tokens | Correct-answer tokens | Aggregate throughput |
| --------------------- | --------: | -----------------: | --------------------: | -------------------: |
| Base                  |     27.6% |              137.2 |                 118.4 |         12,962 tok/s |
| LoRA SFT              |     32.4% |              133.6 |                 107.3 |         13,479 tok/s |
| GRPO correctness      |     34.2% |              133.7 |                 106.8 |         12,846 tok/s |
| GRPO fixed-length     |     32.6% |              128.9 |                 104.0 |         12,115 tok/s |
| GRPO adaptive-latency | **34.4%** |          **132.4** |                 108.9 |         12,733 tok/s |

Key findings:

* SFT improved accuracy by `+4.8 percentage points` over the base model.
* Correctness GRPO added a further `+1.8 percentage points` over SFT.
* Adaptive GRPO achieved the best accuracy, adding `+2.0 percentage points` over SFT.
* Adaptive GRPO improved over the base model by `+6.8 percentage points`.
* Fixed-length GRPO reduced mean output length by `3.5%` versus SFT with nearly unchanged accuracy.
* Adaptive GRPO preserved near-SFT generation length while achieving the highest accuracy.

These results measure post-training and generation efficiency. They do not represent a kernel-level or architectural speedup.

## Methodology

### Data

The GSM8K data is divided into disjoint subsets:

* SFT training set: 1,500 examples
* GRPO training set: 300 examples
* Evaluation set: full GSM8K test split, 1,319 examples

The disjoint training split prevents GRPO from optimizing directly on the SFT examples.

### Stage 1: LoRA SFT

The base model is fine-tuned using parameter-efficient LoRA:

* Base weights loaded in BF16
* LoRA rank: 16
* LoRA applied to all linear layers
* Loss computed only over assistant completion tokens
* One training epoch
* Output: `outputs/sft_lora`

### Stage 2: Verifier-guided GRPO

Each GRPO experiment starts from the same SFT adapter. Candidate completions are sampled in groups and scored using relative rewards.

#### Correctness reward

Rewards exact numerical agreement with the GSM8K reference answer and provides a format bonus for valid structured reasoning.

#### Fixed-length reward

Adds a penalty for excess completion tokens on correctly solved examples:

```text
reward = correctness + format_bonus - fixed_length_penalty
```

#### Adaptive-latency reward

Uses a measured hardware latency model and applies the efficiency penalty conditionally when the sampled group has a sufficiently high solve rate:

```text
latency = intercept + milliseconds_per_token × output_tokens
```

This prevents the model from sacrificing correctness merely to produce shorter answers.

The adaptive reward is an engineering study of hardware-aware reward shaping. It is not presented as a new GRPO algorithm.

## Hardware latency calibration

The SFT model was served through vLLM and benchmarked using:

* 500 requests
* Concurrency: 16
* Maximum completion length: 256 tokens
* P50 end-to-end latency: 919 ms
* P99 end-to-end latency: 3,168 ms
* P50 TTFT: 163 ms
* P50 TPOT: 6.66 ms
* Output throughput: 1,676 tokens/s

The fitted latency model was:

```text
latency_ms = 149.35 + 7.52 × output_tokens
```

Calibration statistics:

```text
R²: 0.482
Latency budget: 919.27 ms
Budget percentile: 50th percentile
Samples: 500
```

The profile should be recalibrated when changing GPU, model, serving engine, batch size, or target concurrency.

## Installation

Use Python 3.10–3.12 with a CUDA-enabled PyTorch installation.

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev]"
```

For vLLM evaluation, use a separate environment if necessary:

```bash
python -m venv /tmp/reasoning-eval
source /tmp/reasoning-eval/bin/activate

pip install -e . --no-deps
pip install "vllm==0.23.0"
```

The training environment should not import vLLM unnecessarily because TRL and vLLM versions can have incompatible integration APIs.

## 1. Prepare the data

```bash
python scripts/prepare_data.py \
  --sft-size 1500 \
  --grpo-size 300 \
  --eval-size 500
```

This creates:

```text
data/sft_train.jsonl
data/grpo_train.jsonl
data/eval.jsonl
data/manifest.json
```

The `eval_size` option controls the local development subset. Final evaluation should use the complete GSM8K test split with `--limit 1319`.

## 2. Train the SFT adapter

```bash
python scripts/train_sft.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir outputs/sft_lora
```

## 3. Merge the SFT adapter for serving

```bash
python scripts/merge_adapter.py \
  --adapter outputs/sft_lora \
  --output-dir outputs/merged_sft
```

## 4. Calibrate latency

Start the vLLM server:

```bash
source /tmp/reasoning-eval/bin/activate

export PYTHONPATH=/marimo/reasoning_grpo
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve outputs/merged_sft \
  --served-model-name efficient-math \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.85
```

In another terminal, benchmark the server:

```bash
python scripts/benchmark_vllm_server.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model efficient-math \
  --tokenizer outputs/merged_sft \
  --concurrency 16 \
  --num-prompts 500 \
  --max-tokens 256 \
  --output-dir results/latency_benchmark
```

Fit the latency model:

```bash
python scripts/fit_latency_profile.py \
  --requests results/latency_benchmark/requests.jsonl \
  --output results/latency_profile.json
```

## 5. Train GRPO variants

Run the experiments sequentially on one GPU:

```bash
python scripts/train_grpo.py \
  --sft-adapter outputs/sft_lora \
  --reward-mode correctness \
  --max-steps 225 \
  --output-dir outputs/grpo_correctness

python scripts/train_grpo.py \
  --sft-adapter outputs/sft_lora \
  --reward-mode fixed \
  --max-steps 225 \
  --output-dir outputs/grpo_fixed

python scripts/train_grpo.py \
  --sft-adapter outputs/sft_lora \
  --reward-mode adaptive \
  --latency-profile results/latency_profile.json \
  --max-steps 225 \
  --output-dir outputs/grpo_adaptive
```

For a smoke test:

```bash
--max-steps 5
```

The GRPO variants are independent and can run in parallel if separate GPUs are available.

## Optional vLLM generation during GRPO

The default trainer uses Transformers generation. To use colocated vLLM generation:

```bash
python scripts/train_grpo.py \
  --sft-adapter outputs/sft_lora \
  --reward-mode correctness \
  --use-vllm \
  --vllm-mode colocate \
  --vllm-gpu-memory-utilization 0.3 \
  --output-dir outputs/grpo_correctness
```

If this causes GPU memory exhaustion, use the default Transformers generation.

## 6. Merge all adapters

```bash
python scripts/merge_adapter.py \
  --adapter outputs/grpo_correctness \
  --output-dir outputs/merged_correctness

python scripts/merge_adapter.py \
  --adapter outputs/grpo_fixed \
  --output-dir outputs/merged_fixed

python scripts/merge_adapter.py \
  --adapter outputs/grpo_adaptive \
  --output-dir outputs/merged_adaptive
```

## 7. Evaluate on the complete GSM8K test set

Run each evaluation in a separate process so the vLLM engine releases GPU memory:

```bash
python scripts/evaluate_vllm.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir results/base \
  --limit 1319

python scripts/evaluate_vllm.py \
  --model outputs/merged_sft \
  --output-dir results/sft \
  --limit 1319

python scripts/evaluate_vllm.py \
  --model outputs/merged_correctness \
  --output-dir results/grpo_correctness \
  --limit 1319

python scripts/evaluate_vllm.py \
  --model outputs/merged_fixed \
  --output-dir results/grpo_fixed \
  --limit 1319

python scripts/evaluate_vllm.py \
  --model outputs/merged_adaptive \
  --output-dir results/grpo_adaptive \
  --limit 1319
```

Summarize the results:

```bash
python scripts/summarize_results.py \
  results/base/summary.json \
  results/sft/summary.json \
  results/grpo_correctness/summary.json \
  results/grpo_fixed/summary.json \
  results/grpo_adaptive/summary.json \
  --output results/comparison.json
```

## Metrics

The evaluation pipeline reports:

* Exact-match GSM8K accuracy
* Mean output tokens
* Median output tokens
* Mean tokens for correct answers
* Aggregate output throughput
* Wall-clock evaluation time

For deployment-oriented evaluation, also measure:

* P50/P99 time to first token
* P50/P99 time per output token
* P50/P99 end-to-end latency
* Request throughput at fixed concurrency
* Output-token throughput
* Peak GPU memory
* LoRA adapter size

Shorter generations can reduce end-to-end latency, but they do not imply faster kernels or lower per-token latency. TTFT and TPOT should remain approximately unchanged for the same architecture and serving configuration.

## Validation

```bash
pytest -q
python -m compileall reasoning_efficiency scripts tests
```

## Project contribution

The project combines:

* LoRA parameter-efficient fine-tuning
* Exact-answer mathematical verification
* Group Relative Policy Optimization
* Length-aware reward shaping
* Hardware-calibrated latency modeling
* vLLM serving benchmarks
* Accuracy–token–latency analysis

The central result is that correctness optimization and efficiency optimization need not be treated independently: a measured serving-cost model can be incorporated into the reward while preserving mathematical accuracy.

## References

* [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)
* [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
* [Hugging Face Open-R1](https://github.com/huggingface/open-r1)
* [DeepSeekMath](https://arxiv.org/abs/2402.03300)
* [vLLM Documentation](https://docs.vllm.ai/)
