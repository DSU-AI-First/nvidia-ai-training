# NVIDIA NeMo Data Designer

_From seed data to quality synthetic datasets._

## What is Data Designer?

NeMo Data Designer helps us generate synthetic datasets using configurable samplers, LLMs, and existing seed data. We can define how data should be generated, preview results, and evaluate the quality of generated records.

## Core workflow

1. **Configure** — Define columns, models, prompts, and generation settings.
2. **Generate** — Create data with or without seed records.
3. **Evaluate** — Validate outputs using programmatic checks and LLM-as-a-Judge.
4. **Export** — Save datasets for downstream training and experimentation.

## Key concepts

- **Seed data:** Existing records used as context or a starting point for generation.
- **LLM columns:** Generate content using prompt templates and configured models.
- **Preview:** Test the configuration and inspect generated examples before creating a dataset.
- **LLM-as-a-Judge:** Score generated outputs against defined criteria using a judge model.
- **Validation:** Check structure, correctness, and quality. Valid configuration does not automatically mean high-quality data.

## Hands-on examples

**1. Without seed data**  
Generate greetings in different languages using sampled input values.

**2. With seed data**  
Enrich product reviews with sentiment labels and concise summaries.

**3. Quality evaluation**  
Use an LLM judge to score summary faithfulness on a 1–5 scale, with a clear rubric for each score.

## Best practices

- Start small and preview before generating at scale.
- Write clear prompts and distinguishable scoring criteria.
- Combine programmatic validation with LLM-based evaluation.
- Inspect failed or low-scoring records rather than trusting scores blindly.
- Keep API keys in Colab Secrets or environment variables—never commit them.

## Learning outcome

By the end, participants should be able to configure a synthetic data workflow, generate structured datasets, and evaluate output quality before using the data downstream.

---

_Training repository: notebooks, datasets, and exercises for hands-on learning._
