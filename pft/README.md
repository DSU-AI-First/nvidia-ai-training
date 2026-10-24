# DPO Fine-Tuning Demo on a T4 GPU (NeMo RL + Qwen2.5-0.5B-Instruct)

An annotated Colab notebook (`DPO.ipynb`) that fine-tunes a small language model with **Direct Preference Optimization (DPO)** on a single Tesla T4, then compares the base and DPO-trained models side by side.

**DPO in one line:** show the model pairs of answers (a *chosen* one and a *rejected* one) and nudge it to make the chosen kind more likely, while a frozen copy of the original model (the *reference*) keeps it from drifting too far.

## Overview

| Item | Value |
|---|---|
| Framework | [NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL) (launched via `uv`) |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Dataset | [`Intel/orca_dpo_pairs`](https://huggingface.co/datasets/Intel/orca_dpo_pairs): 512 train + 88 validation pairs (shuffled, seed 0) |
| Training | 20 steps, global batch size 8, micro batch size 1 |
| Precision | float32 (T4 has no native bf16) |
| Max sequence length | 512 tokens |
| Hardware | Colab **T4 (15 GB) + High-RAM** runtime |

## Requirements

- Google Colab with GPU runtime set to **T4** and **High-RAM**. The standard ~12 GB RAM runtime fails at worker startup (see "Known issues").
- Internet access to clone NeMo RL and download the model and dataset from Hugging Face.

## Notebook walkthrough

1. **Check the hardware**: confirms GPU model, memory, CUDA and PyTorch versions.
2. **Install uv and inspect the default DPO config**: clones NeMo RL and prints `examples/configs/dpo.yaml`.
3. **Prepare the dataset**: writes `train.jsonl` and `val.jsonl` with `prompt`, `chosen`, `rejected` fields.
4. **First successful training run**: 20 DPO steps, no checkpoint saved.
5. **Re-run with checkpoint saving**: same training, saves the model in Hugging Face format.
6. **Compare base vs DPO**: both models answer the same five prompts with greedy decoding.
7. **Read the results**: prints `/content/compare_out.txt`.

To reproduce from scratch: set the runtime to T4 + High-RAM, then run Steps 1, 2, 3, 5, 6, 7 in order. Step 4 is the same training without saving the model.

## Key training command (Step 5)

```bash
cd /content/nemo-rl && MPLBACKEND=Agg uv run examples/run_dpo.py \
  policy.model_name=Qwen/Qwen2.5-0.5B-Instruct \
  policy.tokenizer.name=Qwen/Qwen2.5-0.5B-Instruct \
  policy.precision=float32 \
  policy.train_global_batch_size=8 \
  policy.train_micro_batch_size=1 \
  policy.max_total_sequence_length=512 \
  policy.automodel_cfg.activation_checkpointing=true \
  policy.automodel_cfg.checkpoint.save_consolidated=final \
  dpo.max_num_steps=20 dpo.val_period=10 dpo.val_batches=4 dpo.val_global_batch_size=4 \
  checkpointing.enabled=true checkpointing.save_period=20 \
  checkpointing.save_optimizer=false checkpointing.keep_top_k=1 \
  ++data.train.dataset_name=BinaryPreferenceDataset \
  ++data.train.data_path=/content/train.jsonl ++data.train.split=train \
  ++data.validation.dataset_name=BinaryPreferenceDataset \
  ++data.validation.data_path=/content/val.jsonl ++data.validation.split=train \
  ++data.default.dataset_name=BinaryPreferenceDataset \
  ++data.default.prompt_key=prompt ++data.default.chosen_key=chosen ++data.default.rejected_key=rejected
```

### Flag guide

| Flag | Purpose |
|---|---|
| `policy.precision=float32` | T4 lacks native bf16 |
| `policy.train_global_batch_size=8` | 8 preference pairs per step |
| `policy.train_micro_batch_size=1` | 1 pair per GPU pass to save memory |
| `policy.max_total_sequence_length=512` | Longer examples are skipped |
| `policy.automodel_cfg.activation_checkpointing=true` | Recompute activations to save GPU memory |
| `dpo.max_num_steps=20` | Short demo run |
| `dpo.val_period=10`, `dpo.val_batches=4` | Validate at start and every 10 steps |
| `checkpointing.save_optimizer=false` | Smaller, faster checkpoint |
| `...checkpoint.save_consolidated=final` | Writes a Hugging Face style weights folder |
| `++data.*` | Points NeMo RL at the JSONL files (`++` force-adds keys missing from the default YAML) |
| `MPLBACKEND=Agg` | Avoids the Colab matplotlib backend crash |

## What to watch in the logs

- **Validation loss at step 0 is 0.6931 (= ln 2):** the model cannot yet tell chosen from rejected.
- **`accuracy`:** fraction of pairs where the model prefers the chosen answer (0.91 on validation at step 10).
- **`rewards_chosen_mean` / `rewards_rejected_mean`:** how far the model has moved from the reference on each side. A growing gap means it is learning the preference.
- `dpo.reference_policy_kl_penalty` (default 0.05) is the DPO beta, the "leash" keeping the model near the reference.

## Results

Five prompts, one greedy-decoded answer each:

| Question | Outcome |
|---|---|
| Capital of Australia | DPO shorter, no factual slip (base misplaces Canberra) |
| Why the sky is blue | DPO closer to correct, but both flawed |
| Train average speed | DPO reaches the correct 80 km/h; base is cut off at the 170-token limit |
| Decline a meeting email | DPO drops the odd "As an AI language model" line; both weak |
| Cooked rice overnight | DPO gives one clean correct sentence; base misstates the storage range |

**What changed:** the DPO model is more concise, polished and formal, and follows "briefly" / "two sentences" better.

**What didn't change:** it is not smarter. It is still a 0.5B model that gets some facts wrong. DPO shifted style toward the preferred (longer, more careful, GPT-4-written) answers rather than adding knowledge.

**Caveat:** five questions is a sanity check, not a benchmark. Reward values also drifted a lot over the 20 steps, though outputs stayed coherent.

## Output locations (Colab runtime)

- Trained weights: `/content/nemo-rl/results/dpo/step_20/policy/weights/model/consolidated`
- Side-by-side answers: `/content/compare_out.txt`
- Training logs: `/content/dpo*.log`

These are lost when the runtime is deleted, so download them if you want to keep them.

## Known issues (failed attempts kept in the notebook)

| Problem | Fix |
|---|---|
| Crash on import: Colab's inline matplotlib backend doesn't exist in NeMo RL's isolated env | Prefix command with `MPLBACKEND=Agg` |
| Hydra refuses to override keys not in the default YAML (e.g. `data.train.data_path`) | Use `++` prefix |
| Worker killed by Ray's memory monitor on the standard ~12 GB RAM runtime | Switch to **High-RAM** (~50 GB) |
| Swap file creation fails in Colab (`swapon failed`) | Not possible; use High-RAM |
| Disabling Ray's memory monitor (`RAY_memory_monitor_refresh_ms=0`) doesn't help | Machine is too small; use High-RAM |
| `pkill -f` matched its own command line and killed the cell | Use bracket patterns, e.g. `pkill -9 -f "[r]un_dpo.py"` |
