# PDF → Curated Text Dataset with NVIDIA NeMo Curator

A Google Colab notebook that turns raw PDF files into a clean, quality-filtered text dataset (JSONL) using [NVIDIA NeMo Curator](https://github.com/NVIDIA-NeMo/Curator). The output can be used for LLM training or fine-tuning.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/<your-repo>/blob/main/Nemo_Curator.ipynb)

> Replace `<your-username>` and `<your-repo>` in the badge link above with your GitHub username and repository name.

## Pipeline

```
 PDF files ──► Extract text ──► Split into samples ──► Quality filter & clean ──► Curated JSONL
 (raw_pdf)     (PyMuPDF)        (regex on "Sample N:")  (NeMo Curator pipeline)    (curated/)
```

| Step | What happens | Tool |
|------|--------------|------|
| 1. Setup | Install NeMo Curator, mount Google Drive, create folders | `uv`, `nemo-curator` |
| 2. Extract | Read every PDF and write one JSON record per PDF | PyMuPDF |
| 3. Split | Break the extracted text into individual samples | Python `re` |
| 4. Curate | Filter low-quality text and normalise the rest | NeMo Curator `Pipeline` |
| 5. Inspect | Count and preview the final records | Python `json` |

## Requirements

- A Google account (the notebook stores its data on Google Drive)
- Google Colab (a GPU is **not** required; the notebook installs `nemo-curator[text_cpu]`)
- One or more PDF files with selectable text (scanned images without OCR are skipped)

## How to use

1. Open `Nemo_Curator.ipynb` in Google Colab (use the badge above).
2. Run the install cell and the Drive-mount cell. The folders below are created automatically.
3. Upload your PDF file(s) to `MyDrive/nemo_curator/data/raw_pdf/`.
4. Run the remaining cells in order.
5. Your curated dataset is written to `MyDrive/nemo_curator/data/curated/`.

### Folder layout on Google Drive

```
MyDrive/nemo_curator/
└── data/
    ├── raw_pdf/                    # input: put your PDFs here
    ├── extracted_jsonl/
    │   ├── pdf_extracted.jsonl     # one record per PDF
    │   └── split_extracted/
    │       └── pdf_extracted_split.jsonl   # one record per sample
    └── curated/
        └── pdf_cleaned/            # final curated JSONL
```

## Curation stages

The pipeline reads the split samples, applies the stages below in order, and writes the survivors to JSONL. Filters remove documents; modifiers rewrite text.

| # | Stage | Type | Purpose |
|---|-------|------|---------|
| 1 | `JsonlReader` | Reader | Load the split samples |
| 2 | `WordCountFilter` (10–100,000 words) | Filter | Drop very short or very long documents; saves `word_count` |
| 3 | `UnicodeReformatter` | Modifier | Fix broken Unicode |
| 4 | `NewlineNormalizer` | Modifier | Collapse 3+ newlines into 2 |
| 5 | `NonAlphaNumericFilter` (> 25%) | Filter | Drop symbol-heavy text |
| 6 | `RepeatedLinesFilter` (≥ 0.7 unique) | Filter | Drop repetitive lines |
| 7 | `PunctuationFilter` (> 85%) | Filter | Drop text without sentence endings |
| 8 | `BoilerPlateStringFilter` (> 40%) | Filter | Drop policy/cookie boilerplate |
| 9 | `RepeatingTopNGramsFilter` (n=2, n=3) | Filter | Drop text dominated by one repeated phrase |
| 10 | `RepeatingDuplicateNGramsFilter` (n=5) | Filter | Drop text with many duplicated 5-word sequences |
| 11 | `SymbolsToWordsFilter` | Filter | Drop text with many `#` or `...` tokens |
| 12 | `UrlsFilter` | Filter | Drop link-heavy text |
| 13 | `WordsWithoutAlphabetsFilter` | Filter | Drop text where many "words" have no letters |
| 14 | `MeanWordLengthFilter` (3–10) | Filter | Drop gibberish or glued-together text |
| 15 | `CommonEnglishWordsFilter` | Filter | Drop text that does not look like English prose |
| 16 | `JsonlWriter` | Writer | Save the curated documents |

Stages 9–15 are defined in one `extra_filters` list in the pipeline cell, so a filter can be added, removed or re-tuned by editing that list.

## Output format

Each line of the curated JSONL file is one record:

```json
{"sample_id": 12, "text": "Cleaned sample text ...", "word_count": 124}
```

## Tuning

- **Too many good documents removed:** loosen thresholds (lower `min_words`, raise the ratio limits) or remove a filter from `extra_filters`.
- **Too much noise kept:** tighten thresholds or add filters from `nemo_curator.stages.text.filters`.
- Most filters assume **English** text. For other languages, check each filter's `lang` option or remove the filter.

## Notes and limitations

- The split step expects samples that start with a `Sample N:` marker, and it reads only the first extracted record, so it is written for a single PDF. Adapt the regular expression for other document layouts.
- `BoilerPlateStringFilter` only recognises a fixed list of policy and cookie phrases. Repeated spam text is caught by the repetition filters instead.
- The notebook was developed in Google Colab; paths use `/content/drive/MyDrive/...`.

## Acknowledgements

Built with [NVIDIA NeMo Curator](https://github.com/NVIDIA-NeMo/Curator) and [PyMuPDF](https://pymupdf.readthedocs.io/).
