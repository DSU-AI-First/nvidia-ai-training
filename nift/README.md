# ML Domain-Adaptation Demo — 205 PDFs

**Qwen3-0.6B-Base · NeMo Curator · NeMo AutoModel · One Colab T4**

This notebook performs **causal next-token domain-adaptive pretraining** on ordinary Machine Learning text from 205 PDFs. It does not create instruction/answer pairs.

## Workflow

1. **Extract PDFs** — Parse 205 ML research papers using PyMuPDF
2. **Document-level train/validation split** — Split by paper to avoid leakage
3. **NeMo Curator filtering** — Quality filtering and deduplication
3. **Tokenize** — Prepare data for Qwen3-0.6B-Base
4. **Baseline held-out loss** — Measure pre-training perplexity
5. **NeMo AutoModel QLoRA** — Domain-adaptive pretraining with QLoRA
6. **Post-training loss** — Evaluate perplexity improvement
7. **Save logs/metrics/artifacts** — Persist results to Google Drive

## Key Details

- **Model**: `Qwen/Qwen3-0.6B-Base`
- **Sequence length**: 256 tokens
- **Hardware**: Single T4 GPU (14.56 GB VRAM)
- **Training steps**: 30 (demo configuration)
- **Batch size**: 1 with gradient accumulation × 4
- **Learning rate**: 2e-4
- **Max train blocks**: 6,400
- **Max val blocks**: 160

## Important Notes

> **This notebook is configured for a single T4 and a short faculty demo, not a full production training run.**

- Has not been executed on your Colab account
- NeMo APIs and package compatibility can change
- If the AutoModel stage fails, retain the full traceback rather than rerunning blindly
- No OpenAI API key is used (fully local/NeMo pipeline)

## Data

- **205 PDFs** from ML research papers
- PDFs should be placed in: `/content/drive/MyDrive/DSU-Finetuning_Project/doc/`
- Cached extractions stored in: `/content/drive/MyDrive/DSU-Finetuning_Project/raw/`
- Processed data in: `/content/drive/MyDrive/DSU-Finetuning_Project/data/`
- Artifacts/logs in: `/content/drive/MyDrive/DSU-Finetuning_Project/artifacts/`

## Requirements

```
nemo-automodel
nemo-curator
pymupdf
transformers
accelerate
bitsandbytes
peft
pyyaml
huggingface_hub
torchao>=0.16.0
```

## Hugging Face Authentication

Optional for public model access. Set `HF_TOKEN` in Colab Secrets for authenticated downloads.

## Output

- Training/validation loss curves
- Perplexity metrics (pre and post training)
- Model adapters saved to artifacts directory
- JSONL logs for analysis