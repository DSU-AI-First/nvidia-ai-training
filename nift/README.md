# NeMo Curator + NeMo AutoModel: Pharma PDF → Non-Instruction Fine-Tuning



**Colab lab | CPU for data preparation, optional T4 for LoRA training**

This repository features a NeMo-based workflow rewriting a standard *Non-instruction Causal LLM Fine-Tuning / Domain-Adaptive Continued Pretraining* notebook. It processes the supplied **Metformin-Lipid-Therapy-Knowledge.pdf** along with optional domain extension datasets.

---

## What is genuinely NeMo here?



* **PyMuPDF**: Extracts text from PDFs (NeMo Curator's text API does not replace a dedicated PDF parser).


* **NeMo Curator**: Uses real `Pipeline`, `JsonlReader`, `Modify`, `ScoreFilter`, and `JsonlWriter` components to clean and filter text on the **CPU**.


* **NeMo AutoModel**: Uses `NeMoAutoModelForCausalLM.from_pretrained`, its **LoRA `PeftConfig**`, and the `automodel` recipe to train on a **T4 GPU** using a raw-text next-token dataset factory (**not** instruction-response SFT).


* **Hugging Face/PEFT**: Used for loading saved adapters for inference, as NeMo AutoModel natively supports this checkpoint format.



> **Important Note:** A small multi-page PDF alone is insufficient to build a robust medical-domain model, so the workflow optionally incorporates public study texts (via generated study PDFs) to expand the learning experiment. This is designed strictly for architectural exploration and workflow practice, not clinical advice.
> 
> 

---

## What You Will Learn



1. How a real **NeMo Curator** pipeline cleans and filters text on the CPU.


2. How raw text transforms into next-token training examples via `input_ids` and target labels **shifted by one**.


3. How a single **NeMo AutoModel YAML recipe** manages an end-to-end LoRA run.


4. How to analyze live training logs and evaluate model quality using **perplexity on training vs. held-out validation text**.



---

## Files Created Under `/content/nemo_pharma_lab`

* `raw/pdf_pages.jsonl`: One JSON record per extracted PDF page.


* `curated/`: Cleaned text output shards from NeMo Curator.


* `dataset/train.jsonl`, `dataset/validation.jsonl`: Page-level dataset splits.


* `pharma_causal_dataset.py`: Custom dataset factory referenced by the YAML recipe.


* `pharma_t4_lora.yaml`: The declarative training configuration.


* `tinyllama_safetensors/`: Base model files.


* `checkpoints/`: Trained LoRA adapter checkpoints.


* `train.log`: Full, unfiltered execution logs.



---

## Quickstart & Execution Overview



1. **Environment Setup:** Open the notebook in Google Colab, select a **CPU runtime** for text cleaning or a **T4 GPU** if running training.


2. **Data Curation:** Run the text extraction and NeMo Curator pipeline stages to clean unicode artifacts, normalize spacing, and filter records.


3. **Dataset Split:** Build page-level causal token blocks ensuring training and validation splits remain distinct.


4. **Training:** Execute the NeMo AutoModel CLI command using the configuration recipe:


```bash
automodel pharma_t4_lora.yaml

```


5. **Evaluation:** Compute perplexity metrics and perform text generation comparisons between the base model and the fine-tuned LoRA adapter.



---

## NVIDIA References



* [NeMo Curator Text Cleaning Documentation](https://docs.nvidia.com/nemo/curator/v26.09/curate-text/process-data/content-processing/text-cleaning)

* [NeMo Curator Quality Assessment Documentation](https://docs.nvidia.com/nemo/curator/v26.09/curate-text/process-data/quality-assessment)

* [NeMo AutoModel SFT & PEFT Recipes](https://www.google.com/search?q=https://docs.nvidia.com/nemo/automodel/latest/recipes-e2e-examples/sft-peft)

* [NeMo AutoModel Custom Datasets](https://www.google.com/search?q=https://docs.nvidia.com/nemo/automodel/latest/datasets/text-dataset)
