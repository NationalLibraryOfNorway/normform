# NormForm (`v0.1.0` — Experimental)

> [!WARNING]
> **Experimental Release (`v0.1.0`)**: NormForm and its accompanying neural models and style profiles are currently **experimental (`v0.1`)**. CLI flags, Python APIs, model checkpoints, and style cache schemas may evolve in upcoming releases. Always review standardized outputs in production workflows.

**NormForm** standardizes Norwegian Bokmål (`nob`) and Nynorsk (`nno`) text by making consistent choices among contextually valid morphological alternatives, while guaranteeing that non-alternative text is never altered.

It also generates `[alt1|alt2]` morphological alternative brackets for ASR evaluation ([AltMetrics](https://github.com/peregilk/altmetrics)), audits documents for internal morphological inconsistencies, extracts representative style steering words from reference texts, and runs high-throughput streaming corpus standardization on GPU.

---

## Installation

```bash
pip install normform
# Or with batch dataset extras (Parquet / Hugging Face Datasets):
pip install "normform[batch]"
```

> [!NOTE]
> NormForm uses the public pinned two-task neural models (`NbAiLab/normform-bokmaal-multitask` and `NbAiLab/normform-nynorsk-multitask`) and public reviewed Hugging Face style caches (`NbAiLab/normform-bokmaal-redaksjonsnorm` and `NbAiLab/normform-nynorsk-nasjonalbiblioteknorm`). It **does not** require an Ordbank API key or Hugging Face authentication for public models and styles.

---

## Quick Start

### 1. Command Line (`normform`)

```bash
# Standardize a sentence using the official Bokmål Redaksjonsnorm style:
normform standardize -t "Jenta kasta ballen i boken." --lang nob --style redaksjonsnorm

# Standardize a Nynorsk sentence using Nasjonalbiblioteknorm:
normform standardize -t "Me køyrde heim på laurdag." --lang nno --style nasjonalbiblioteknorm

# Standardize using custom steering prompt words:
normform standardize -t "Jenta hoppa høyt." -p "jenta,kasta,boka"

# Generate [alt1|alt2] morphological alternative brackets:
normform alternatives -t "Katta ligger på matta." --lang nob

# Audit a document for internal morphological inconsistencies (fast raw check):
normform audit -i document.txt --raw

# Extract top 10 representative style prompt words from a reference text:
normform extract-style -i reference_document.txt --lang nob -n 10

# High-throughput 1-GPU streaming corpus standardization (.jsonl, .parquet, or HF dataset):
normform batch -i corpus.jsonl -o corpus_standardized.jsonl \
  --lang nob --style NbAiLab/normform-bokmaal-redaksjonsnorm --preserve-unreviewed
```

### 2. Python API

```python
from normform import Standardizer

# Initialize with a built-in style alias, Hugging Face repo ID, or local .yaml file:
std = Standardizer(lang="nob", style="redaksjonsnorm")

# Standardize text:
print(std("Katta ligger på matten."))

# Expand into [alt1|alt2] morphological alternative brackets:
print(std.alternatives("Katta ligger på matta."))
```

---

## Managing & Sharing Styles on Hugging Face (`normform styles`)

Each NormForm style lives in its own Hugging Face dataset repository (containing `style.yaml` and a `README.md` style card).

```bash
# List official built-in styles and aliases:
normform styles list

# Pull an official style locally to inspect or customize it:
normform styles pull redaksjonsnorm -o my_custom_style.yaml

# Edit my_custom_style.yaml, then push it to your own private HF repo:
normform styles push -f my_custom_style.yaml --repo my-username/my-normform-style

# Use your custom HF style anywhere:
normform standardize -i document.txt --style my-username/my-normform-style
```

### Official Pinned Models & Style Repositories

| Language | Two-Task Neural Model | Official Style Repository | Short Aliases |
| :--- | :--- | :--- | :--- |
| **Bokmål (`nob`)** | `NbAiLab/normform-bokmaal-multitask@eb83a85c52bea137e2bbb3ba95e29be3e9476923` | `NbAiLab/normform-bokmaal-redaksjonsnorm` | `redaksjonsnorm`, `redaksjonsstil`, `bokmaal` |
| **Nynorsk (`nno`)** | `NbAiLab/normform-nynorsk-multitask@350cf9d3aeeba2f390353f4147ef2910dff7b0ee` | `NbAiLab/normform-nynorsk-nasjonalbiblioteknorm` | `nasjonalbiblioteknorm`, `nasjonalbibliotekstil`, `nynorsk` |

---

## License

Apache License 2.0. Created by Per Egil Kummervold and Magnus Breder Birkenes at the National Library of Norway (Nasjonalbiblioteket).
