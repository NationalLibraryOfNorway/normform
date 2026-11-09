# NormForm (`v0.1` — Experimental)

> [!WARNING]
> **Experimental Release (`v0.1`)**: NormForm and its accompanying neural models and style profiles are currently **experimental**. CLI flags, Python APIs, model checkpoints, and style cache schemas may evolve in upcoming releases. Always review standardized outputs in production workflows.

**NormForm** is a Python toolkit and command-line interface for **context-aware Norwegian morphological standardization, alternative generation, and style auditing** in both **Bokmål (`nob`)** and **Nynorsk (`nno`)**.

Written Norwegian officially permits wide intra-norm morphological and orthographic variation—such as feminine vs. common gender nouns (`boka` / `boken`, `klokka` / `klokken`), weak preterite endings (`kasta` / `kastet`), infinitive endings in Nynorsk (`bruka` / `bruke`, `gjera` / `gjere`), and lexical spelling variants (`fram` / `frem`, `sjukehuset` / `sykehuset`, `fyrste` / `første`). NormForm lets you:

1. **Standardize text or large corpora** to a consistent sub-norm style (such as *Bokmål Redaksjonsnorm*, *Nynorsk Nasjonalbiblioteknorm*, or your own custom style profile).
2. **Generate `[alt1|alt2]` morphological alternative brackets** for speech recognition (ASR) and text generation evaluation with [`altmetrics`](https://pypi.org/project/altmetrics/) and [`altwer`](https://pypi.org/project/altwer/).
3. **Audit documents and corpora** for internal morphological inconsistencies or rank the exact morphological changes made across a standardized dataset.
4. **Extract empirical style prompt words** from any reference document using information-theoretic submodular maximization.
5. **Share, fork, and version style profiles** as lightweight YAML repositories on Hugging Face (`1 repo = 1 style`).

---

## Table of Contents

1. [What NormForm Does — and What It Does NOT Do](#1-what-normform-does--and-what-it-does-not-do)
2. [Installation](#2-installation)
3. [Quick Start (CLI & Python API)](#3-quick-start-cli--python-api)
4. [Feature Guide](#4-feature-guide)
   - [4.1 Standardizing Text & Documents (`normform standardize`)](#41-standardizing-text--documents-normform-standardize)
   - [4.2 High-Throughput 1-GPU Batch Corpus Standardization (`normform batch`)](#42-high-throughput-1-gpu-batch-corpus-standardization-normform-batch)
   - [4.3 Generating `[alt1|alt2]` Morphological Alternatives (`normform alternatives`)](#43-generating-alt1alt2-morphological-alternatives-normform-alternatives)
   - [4.4 Auditing Inconsistencies & Change Statistics (`normform audit`)](#44-auditing-inconsistencies--change-statistics-normform-audit)
   - [4.5 Extracting Steering Style Words (`normform extract-style`)](#45-extracting-steering-style-words-normform-extract-style)
   - [4.6 Managing & Customizing Styles (`normform styles`)](#46-managing--customizing-styles-normform-styles)
5. [Style File Format (`style.yaml`)](#5-style-file-format-styleyaml)
6. [Official Models & Style Repositories](#6-official-models--style-repositories)
7. [License & Credits](#7-license--credits)

---

## 1. What NormForm Does — and What It Does NOT Do

NormForm is built around a **strict non-destructive lattice contract** ([`normform.contract`](src/normform/contract.py)). The neural model is never allowed to freely rewrite your text. Instead, every sentence is parsed into a verified template of literal spans and `[alt1|alt2]` candidate slots (`Template.align`), and only validated candidate slots can change.

### What NormForm DOES:
- **Chooses consistently among valid intra-norm alternatives in context**: e.g., in *Bokmål Redaksjonsnorm*, selecting `jenta` and `boka` for concrete feminine nouns, `klokken` for `klokka | klokken`, `kastet` for weak preterites (`kasta | kastet`), and `utafor` / `sjukehuset` for lexical variants; or in *Nynorsk Nasjonalbiblioteknorm*, selecting `-a` infinitives (`bruka`, `gjera`, `vera`), `vi`, `fyrste`, and `sundagen`.
- **Guarantees byte-for-byte preservation of all non-alternative text**: punctuation, whitespace, line breaks, numbers, dates, URLs, email addresses, proper names, and out-of-vocabulary words are never altered.
- **Preserves source token casing automatically**:
  - Lowercase: `jenten` $\rightarrow$ `jenta`
  - Titlecase (sentence-initial): `Jenten` $\rightarrow$ `Jenta`
  - Uppercase: `JENTEN` $\rightarrow$ `JENTA`
  - Mixed internal capitals: `vebb-TV` $\rightarrow$ `web-TV`
- **Protects homographs and semantic collisions (`invalid_pairs`)**: When two surface forms can overlap in inflection tables but differ in meaning or grammatical role (e.g., `verken | verket` *"neither"* vs. *"the pain/work"*, `bad | badet`, `slott | slåss`), the style profile's `invalid_pairs` guard forces NormForm to preserve the original token unchanged.
- **Maintains document- and corpus-level consistency**: Once a candidate group (`boka | boken`) is resolved—either from a reviewed style cache or by the neural style model—that exact decision is cached and applied consistently across every occurrence.

### What NormForm Does NOT Do:
- **Does NOT translate between Bokmål and Nynorsk**: Specify `--lang nob` for Bokmål text and `--lang nno` for Nynorsk text.
- **Does NOT fix general typos, grammar errors, or punctuation**: Words that are not valid Norwegian forms with morphological alternatives are passed through untouched.
- **Does NOT paraphrase, reorder words, or insert/delete tokens**: Every output sentence is guaranteed to align 1-to-1 with the source sentence's literal template.
- **Does NOT rewrite syntactic agreement across neighboring tokens**: NormForm substitutes individual morphological word slots (`[boka|boken]`). It does not rewrite surrounding articles or adjectives outside the candidate slot (e.g., if a source phrase has `ei lita bok`, individual word slots are only changed where each slot's own paradigm allows it).
- **Does NOT use brittle suffix-stripping heuristics on compounds**: Novel compound words absent from the seed cache are either resolved in full sentential context by the neural model or preserved unchanged when `--preserve-unreviewed` is enabled.

### Concrete Examples: What Changes vs. What Stays Untouched

| Scenario | Input Text | Output (`--style redaksjonsnorm` / `nasjonalbiblioteknorm`) | Why? |
| :--- | :--- | :--- | :--- |
| **Bokmål `redaksjonsnorm`** | `"Jenten kasta boken og så på klokka utenfor sykehuset."` | `"Jenta kastet boka og så på klokken utafor sjukehuset."` | `redaksjonsnorm` prefers `-a` on concrete feminine nouns (`jenta`, `boka`), `-et` in weak past tense (`kastet`), `klokken`, `utafor`, and `sjukehuset`. |
| **Nynorsk `nasjonalbiblioteknorm`** | `"Me ønskjer å bruke den første søndagen til å gjere dette."` | `"Vi ynskjer å bruka den fyrste sundagen til å gjera dette."` | `nasjonalbiblioteknorm` uses `-a` infinitives (`bruka`, `gjera`), `vi`, `ynskjer`, `fyrste`, and `sundagen`. |
| **Homograph / collision guard (`invalid_pairs`)** | `"Han tok verken mat eller drikke."` | `"Han tok verken mat eller drikke."` *(unchanged)* | `verken \| verket` is listed in `invalid_pairs` (conjunction *"neither"* vs. noun *"the pain/work"*), so NormForm never alters it. |
| **Cross-token agreement boundary** | `"Hun leste ei lita bok."` | `"Hun leste ei lita bok."` *(unchanged)* | Indefinite singular `bok` has no alternative slot, and NormForm never rewrites surrounding articles/adjectives (`ei lita` $\not\rightarrow$ `en liten`) across token boundaries. |
| **Cross-language input (Nynorsk given to `--lang nob`)** | `"Eg gjekk heim frå skulen."` | `"Eg gjekk heim frå skulen."` *(unchanged)* | NormForm is not a Nynorsk$\leftrightarrow$Bokmål translator; non-target-norm words have no valid target-norm alternative brackets. |
| **Typos, proper nouns, URLs & dates** | `"Per-Egil besøkte https://nb.no den 17. mai i Осло (skrivefeil: bokk)." ` | `"Per-Egil besøkte https://nb.no den 17. mai i Осло (skrivefeil: bokk)."` *(unchanged)* | Out-of-vocabulary words, typos (`bokk`), names, URLs, and numbers are locked outside `Template` slots and preserved byte-for-byte. |

---

## 2. Installation

```bash
pip install normform

# Or install with batch dataset extras (Parquet, Hugging Face Datasets, progress bars):
pip install "normform[batch]"
```

**Requirements:**
- Python $\ge 3.10$ (tested on Python 3.12 with PyTorch $\ge 2.0$ and Transformers $\ge 4.40$).
- A CUDA-capable GPU is strongly recommended for model inference and corpus batch processing (`--device cuda:0` or `--device auto`). CPU execution is supported for small texts and `--raw` audits.
- **No API keys required**: The pinned neural models and official style repositories on Hugging Face are public and downloaded automatically on first use.

---

## 3. Quick Start (CLI & Python API)

### Command Line (`normform`)

```bash
# Check version
normform --version

# 1. Standardize a Bokmål sentence using the official Redaksjonsnorm style:
normform standardize -t "Jenten kasta boken og så på klokka utenfor sykehuset." --lang nob --style redaksjonsnorm
# -> Jenta kastet boka og så på klokken utafor sjukehuset.

# 2. Standardize a Nynorsk sentence using Nasjonalbiblioteknorm:
normform standardize -t "Me ønskjer å bruke den første søndagen til å gjere dette." --lang nno --style nasjonalbiblioteknorm
# -> Vi ynskjer å bruka den fyrste sundagen til å gjera dette.

# 3. Generate [alt1|alt2] morphological alternative brackets:
normform alternatives -t "Katta ligger på matta." --lang nob
# -> [Katta|Katten] ligger på [matta|matten].

# 4. Check a document for internal sub-norm inconsistencies (without modifying it):
normform audit -i manuscript.txt --lang nob --raw
```

### Python API

```python
from normform import Standardizer

# Initialize with a built-in style alias, Hugging Face repo ID, or local .yaml file:
std = Standardizer(
    lang="nob",
    style="redaksjonsnorm",       # or "NbAiLab/normform-bokmaal-redaksjonsnorm"
    preserve_unreviewed=False,    # True = only apply pre-reviewed cache entries
    device="auto",
)

# 1. Standardize text (preserves casing, punctuation, and layout):
text = std("Jenten kasta boken og så på klokka utenfor sykehuset.")
print(text)
# -> "Jenta kastet boka og så på klokken utafor sjukehuset."

# 2. Generate [alt1|alt2] morphological alternative brackets:
bracketed = std.alternatives("Katta ligger på matta.")
print(bracketed)
# -> "[Katta|Katten] ligger på [matta|matten]."

# 3. Check a text or document for internal sub-norm inconsistencies:
issues = std.audit("Boka lå på bordet. Senere leste jeg boken.", raw=True)
for issue in issues:
    print(issue["candidate_group"], "->", issue["first_occurrence"]["choice"], "vs", issue["inconsistent_occurrence"]["choice"])
# -> ['boka', 'boken'] -> Boka vs boken
```

---

## 4. Feature Guide

### 4.1 Standardizing Text & Documents (`normform standardize`)

Use `normform standardize` for interactive strings, plain-text documents (`.txt`), or small JSONL files (`.jsonl`).

#### Choosing Your Standardization Mode

NormForm supports three ways to control how morphological choices are made:

| Mode | CLI Flags | How Seen Words Are Handled | How Unseen Compounds Are Handled | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **1. Strict Reviewed Mode** | `--style <style> --preserve-unreviewed` | Uses the human-reviewed choices in `style.yaml` (33k+ Bokmål / 12k+ Nynorsk entries). | **Preserved unchanged** (never calls the style model on unreviewed words). | Strict editorial publishing & zero-surprise corpus runs. |
| **2. Hybrid Style + Neural Extrapolation** *(Default with `--style`)* | `--style <style>` | Uses the human-reviewed choices in `style.yaml` in $O(1)$ lookup. | Resolved in context by the neural style model (steered by the style's `prompt_words`) and cached for the rest of the run. | Full-coverage corpus standardization where rare compounds should also follow the target style. |
| **3. Zero-Shot Prompt Steering** | `-p "word1,word2,..."` *(no `--style`)* | Resolved dynamically by the neural model steered by your prompt words and cached per document/run. | Resolved dynamically by the neural model and cached per document/run. | Quick experimentation or matching an ad-hoc style from 5–10 indicator words. |

#### Examples

```bash
# Standardize a .txt file with Redaksjonsnorm and write to output.txt:
normform standardize -i article.txt -o article_standardized.txt --lang nob --style redaksjonsnorm

# Strict reviewed-only mode (never invoke the style model on unseen compounds;
# only apply the 33,221 human/linguist-reviewed choices in the style cache):
normform standardize -i article.txt -o article_standardized.txt \
  --lang nob --style redaksjonsnorm --preserve-unreviewed

# Standardize toward a conservative Bokmål profile without a style file by supplying steering prompt words:
normform standardize -t "Jenta kasta boka på hylla." \
  --lang nob -p "jenten,kastet,boken,hyllen,frem"
# -> Jenten kastet boken på hyllen.

# Persist choices across multiple runs in a local JSON cache and write a Markdown summary:
normform standardize -i chapter1.txt -o chapter1_norm.txt \
  --lang nob --style redaksjonsnorm \
  --cache project_choices.json --summary-output chapter1_report.md
```

---

### 4.2 High-Throughput 1-GPU Batch Corpus Standardization (`normform batch`)

For large datasets (tens of thousands to millions of records in `.jsonl`, `.parquet`, directory shards, or Hugging Face datasets), use **`normform batch`**.

#### How the 3-Stage Batch Engine Works
1. **Background I/O Prefetch & Sentence Splitting**: A background producer thread streams records from disk or Hugging Face and splits text into whitespace-preserving sentence spans so disk/network I/O overlaps with GPU compute.
2. **Stage 1 — Length-Bucketed GPU Alternative Expansion**: Non-blank sentences in each chunk (`--chunk-size 4096`) are sorted by character length to minimize padding waste, expanded in large GPU batches (`--batch-size 512`), and validated once with `clean_direct` + `parse` + `Template.align`. Over-capacity outlier sentences are automatically isolated via binary bisection and preserved unchanged.
3. **Stage 2 — Greedy Set-Cover & Context-Style Resolution**:
   - All candidate groups already present in your `--style` cache are resolved in $O(1)$ memory lookup with zero extra GPU calls.
   - If unseen candidate groups appear in the chunk (and `--preserve-unreviewed` is not set), NormForm runs **Greedy Set-Cover** (`greedy_set_cover`) to pick the *minimum subset of sentences* needed to cover all unseen candidate groups, runs `adapter.style_batch()` on just those sentences, and caches the resulting choices for the rest of the corpus.
4. **Stage 3 — Async Streaming Rewrite & Sealed Cache Export**: Sentences are rendered using the updated cache, reassembled into records, and written asynchronously to `.jsonl`, `.parquet` (Zstandard-compressed), or `.txt`. At completion, NormForm exports a human-editable **`*_sealed_cache.yaml`** top-sorted by actual corpus change impact (`# seen: ..., changed: ..., already_correct: ..., change_rate: ...%`).

#### Batch Examples

```bash
# 1. Standardize a large JSONL corpus on GPU 0 using Bokmål Redaksjonsnorm:
normform batch \
  -i /path/to/corpus.jsonl \
  -o /path/to/corpus_standardized.jsonl \
  --lang nob \
  --style NbAiLab/normform-bokmaal-redaksjonsnorm \
  --text-field text \
  --output-field normalized \
  --batch-size 512 \
  --chunk-size 4096 \
  --device cuda:0

# 2. Standardize a Parquet file or directory of Parquet shards in strict reviewed-only mode:
normform batch \
  -i /path/to/parquet_shards/ \
  -o /path/to/standardized.parquet \
  --lang nno \
  --style NbAiLab/normform-nynorsk-nasjonalbiblioteknorm \
  --preserve-unreviewed \
  --sealed-cache-output /path/to/nynorsk_run_sealed_cache.yaml

# 3. Stream directly from a Hugging Face dataset repository:
normform batch \
  -i hf://NbAiLab/npsc_orto \
  --dataset-split train \
  -o npsc_standardized.jsonl \
  --lang nob \
  --style redaksjonsnorm \
  --limit 10000
```

#### Inspecting the Sealed Cache (`*_sealed_cache.yaml`) & Iterative Refinement

Every `normform batch` run automatically writes a `*_sealed_cache.yaml` file sorted by how many times each morphological choice actually modified tokens in your corpus (`changed` descending, then `seen` descending):

```yaml
_metadata:
  description: NormForm Sealed Morphological Style Cache
  total_entries: 33412
  seed_entries: 33221
  newly_discovered: 191
  prompt_words: [jenta, boka, klokka, tida, uka, lista, kasta, skada, fram, sju]
  sort_order: changed_desc_then_seen_desc
choices:
  boka | boken: boka  # seen: 1842, changed: 1410, already_correct: 432, change_rate: 76.5% [seed]
  kasta | kastet: kastet  # seen: 920, changed: 615, already_correct: 305, change_rate: 66.8% [seed]
  klokka | klokken: klokken  # seen: 540, changed: 488, already_correct: 52, change_rate: 90.4% [seed]
   Spesialboka | Spesialboken: spesialboka  # seen: 14, changed: 11, already_correct: 3, change_rate: 78.6% [NEW]
```

**Recommended Corpus Curation Workflow ("Run $\rightarrow$ Inspect $\rightarrow$ Lock"):**
1. Run `normform batch` on your corpus (or a 10,000-record sample with `--limit 10000`).
2. Open `*_sealed_cache.yaml` and search for `[NEW]` to inspect newly discovered compound words, or review the top rows with the highest `changed:` counts.
3. Edit any choice in `*_sealed_cache.yaml` (or move a domain-specific collision into `invalid_pairs`), and re-run with `--style my_reviewed_sealed_cache.yaml --preserve-unreviewed` for 100% deterministic corpus standardization.

#### Python API for Batch Standardization

```python
from pathlib import Path
from normform import run_batch_standardization

run_batch_standardization(
    input_source="/path/to/corpus.jsonl",
    output_path=Path("/path/to/corpus_standardized.jsonl"),
    sealed_cache_output=Path("/path/to/sealed_cache.yaml"),
    lang="nob",
    style="redaksjonsnorm",
    preserve_unreviewed=False,
    text_field="text",
    output_field="normalized",
    batch_size=512,
    chunk_size=4096,
    device="cuda:0",
)
```

---

### 4.3 Generating `[alt1|alt2]` Morphological Alternatives (`normform alternatives`)

`normform alternatives` expands each sentence into alphabetically sorted `[alt1|alt2]` brackets while keeping all non-alternative text unchanged. This output is directly compatible with [`altmetrics`](https://pypi.org/project/altmetrics/) and [`altwer`](https://pypi.org/project/altwer/) for ASR/TTS evaluation where any valid spelling in the bracket is accepted as correct.

```bash
# Single sentence:
normform alternatives -t "Jenta kasta ballen." --lang nob
# -> [Jenta|Jenten] [kasta|kastet] ballen.

# Whole file (.txt or .jsonl):
normform alternatives -i references.jsonl -o references_with_alternatives.jsonl --lang nob
```

---

### 4.4 Auditing Internal Sub-Norm Consistency & Change Statistics (`normform audit`)

A very common use case for NormForm is **checking whether a document, book, or corpus is internally consistent in its sub-norm choices**—without rewriting the text.

For example, an author or transcription dataset might unintentionally mix `boka` and `boken`, `fram` and `frem`, or `kasta` and `kastet` across different paragraphs. Because `normform audit` compares morphological choices case-insensitively across contextually validated `[alt1|alt2]` slots, sentence-initial capitalization (`Boka` vs. `boka`) is treated as consistent, while actual sub-norm switches (`Boka` vs. `boken`) are flagged with exact record and sentence locations.

#### A. Check a Raw Document or Corpus for Internal Sub-Norm Inconsistencies (`--raw`)

```bash
# Human-readable terminal report of all internal sub-norm inconsistencies:
normform audit -i manuscript.txt --lang nob --raw
```

Example terminal output:
```text
Found 2 internal sub-norm inconsistencies in manuscript.txt:

  1. [boka | boken]
     - First choice : 'Boka' (record 0, sentence 0: "Boka lå på bordet hele dagen.")
     - Conflict     : 'boken' (record 0, sentence 2: "Senere på kvelden leste jeg boken.")

  2. [fram | frem]
     - First choice : 'fram' (record 1, sentence 0: "Vi gikk fram til scenen.")
     - Conflict     : 'frem' (record 4, sentence 1: "Hun kom frem i tide.")
```

Export a structured JSON report for automated QA pipelines:
```bash
normform audit -i corpus.jsonl --lang nob --raw --json -o inconsistencies.json
```

Example JSON structure (`inconsistencies.json`):
```json
[
  {
    "candidate_group": ["boka", "boken"],
    "first_occurrence": {
      "record_index": 0,
      "sentence_index": 0,
      "choice": "Boka",
      "sentence": "Boka lå på bordet hele dagen."
    },
    "inconsistent_occurrence": {
      "record_index": 0,
      "sentence_index": 2,
      "choice": "boken",
      "sentence": "Senere på kvelden leste jeg boken."
    }
  }
]
```

In Python:
```python
from normform import Standardizer

std = Standardizer(lang="nob")
issues = std.audit("Boka lå på bordet. Jeg leste boka. Senere tok jeg boken.", raw=True)
# Returns the same list of dicts with 'candidate_group', 'first_occurrence', and 'inconsistent_occurrence'
```

#### B. Model & Style Consistency Audit (`--uncached`)
Tests whether the neural model (under a given `--style` or `--prompt`) makes conflicting choices for the same candidate group across different sentences when cross-sentence caching is disabled:

```bash
normform audit -i evaluation_set.jsonl --lang nob -p "boka,jenta,kasta" --uncached -o audit.json
```

#### C. Post-Standardization Corpus Change Statistics (`--stats`)
Compares the source `--text-field` and standardized `--normalized-field` in a processed `.jsonl` or `.parquet` corpus and outputs a ranked YAML report of the top morphological groups that caused changes:

```bash
normform audit -i corpus_standardized.jsonl --stats --top 50 -o change_report.yaml
```

Example output:
```yaml
corpus: corpus_standardized.jsonl
total_records_scanned: 50000
total_records_changed: 14210 (28.42%)
unique_alternative_groups_changed: 842
top_alternatives:
  boka | boken: {chosen: boka, total_changes: 1410, changed_from: {boken: 1410}}
  kasta | kastet: {chosen: kastet, total_changes: 615, changed_from: {kasta: 615}}
  klokka | klokken: {chosen: klokken, total_changes: 488, changed_from: {klokka: 488}}
```

---

### 4.5 Extracting Steering Style Words (`normform extract-style`)

Given an existing reference document (for example, an editorial style sample or an author's manuscript), `normform extract-style` discovers the top $N$ single-word morphological forms that best explain the document's empirical style choices via **Information-Theoretic Log-Likelihood Submodular Maximization**:

```bash
# Output full JSON report with Explained Style Variance (%):
normform extract-style -i reference_article.txt --lang nob -n 10

# Output only the comma-separated prompt string for direct piping:
PROMPTS=$(normform extract-style -i reference_article.txt --lang nob -n 10 --prompt-only)
normform standardize -i new_article.txt -p "$PROMPTS" --lang nob
```

---

### 4.6 Managing & Customizing Styles (`normform styles`)

NormForm decouples **code** from **linguistic styles**. Every style profile is a self-contained YAML file (`style.yaml`) hosted in its own Hugging Face dataset repository (`1 repo = 1 style`).

```bash
# 1. List built-in official styles and short aliases:
normform styles list

# 2. Download an official style to a local YAML file to inspect or customize:
normform styles pull redaksjonsnorm -o my_style.yaml

# 3. Edit my_style.yaml in any text editor, then push to your own HF dataset repo:
#    (Creates a private HF dataset repo by default; add --public to make it public)
normform styles push -f my_style.yaml --repo username/my-normform-style --lang nob

# 4. Use your custom style from anywhere (or pin a specific commit with @<sha>):
normform standardize -i document.txt --style username/my-normform-style
normform batch -i corpus.jsonl --style username/my-normform-style@main
```

---

## 5. Style File Format (`style.yaml`)

A NormForm `style.yaml` file has three human-readable sections:

```yaml
_metadata:
  language: nob
  style_name: my-custom-style
  prompt_words:
    - jenta
    - boka
    - kastet
    - fram
    - sjukehuset

choices:
  # Mapping from "alt1 | alt2" (lowercase, alphabetical) to the chosen lowercase form:
  boka | boken: boka
  fram | frem: fram
  jenta | jenten: jenta
  kasta | kastet: kastet
  klokka | klokken: klokken
  "no | nå": nå

invalid_pairs:
  # Ambiguous or homographic candidate groups that must NEVER be altered:
  bad | badet: Homographic noun/verb inflection; preserve source token.
  verken | verket: Distinct lemmas (conjunction 'verken' vs. noun 'verket').
```

> [!TIP]
> NormForm includes a custom YAML loader ([`NormFormYamlLoader`](src/normform/styles.py)) that prevents YAML 1.1 from turning Norwegian words like `no` / `NO` (*"now"* in Nynorsk) into boolean `False`, even if unquoted.

---

## 6. Official Models & Style Repositories

All official models and style repositories are publicly available on Hugging Face under [`NbAiLab`](https://huggingface.co/NbAiLab) and pinned by commit SHA for reproducibility:

| Language | Two-Task Neural Model (`T5Gemma2`) | Official Style Dataset Repository | Built-in CLI Aliases |
| :--- | :--- | :--- | :--- |
| **Bokmål (`nob`)** | [`NbAiLab/normform-bokmaal-multitask`](https://huggingface.co/NbAiLab/normform-bokmaal-multitask)<br>`@bdd18aac37718bd1d313f7adbaada7366ca84d12` | [`NbAiLab/normform-bokmaal-redaksjonsnorm`](https://huggingface.co/datasets/NbAiLab/normform-bokmaal-redaksjonsnorm)<br>`@8542693a8e5fe5e7294e0cee1c87019344bec03d` | `redaksjonsnorm`, `redaksjonsstil`, `bokmaal-redaksjonsnorm`, `bokmaal`, `nob` |
| **Nynorsk (`nno`)** | [`NbAiLab/normform-nynorsk-multitask`](https://huggingface.co/NbAiLab/normform-nynorsk-multitask)<br>`@1453a74d1d7ba8e9816d69a30c960437a0923484` | [`NbAiLab/normform-nynorsk-nasjonalbiblioteknorm`](https://huggingface.co/datasets/NbAiLab/normform-nynorsk-nasjonalbiblioteknorm)<br>`@ceaa918476e2ffcba477964922b580c1c19feb71` | `nasjonalbiblioteknorm`, `nasjonalbibliotekstil`, `nynorsk-nasjonalbiblioteknorm`, `nynorsk`, `nno` |

### Linguistic Summary of the Official Style Profiles

- **Bokmål — `NbAiLab/normform-bokmaal-redaksjonsnorm` (33,221 reviewed choices, 200 `invalid_pairs`)**:
  - **Feminine nouns**: Uses `-a` for concrete/everyday feminine nouns (`boka`, `jenta`, `hylla`, `katta`, `tida`, `uka`, `lista`, `helga`), while selecting `-en` for specific editorial nouns (`klokken`, `avisen`, `gruppen`, `grensen`).
  - **Verbs**: Uses `-et` in the preterite and past participle of weak `-e` verbs (`kastet`, `laget`, `skadet`, `snakket`) and `stod` (`sto | stod`).
  - **Neuter plurals**: Uses short indefinite plural (`produkt`, `resultat`, `prosjekt`, `dokument`) and `-ene` in definite plural (`husene`, `årene`, `barna`).
  - **Lexical variants**: Uses `fram` / `framover` / `framtida`, `utafor`, `innenfor`, `sju` / `tjuende`, `sjukehuset`, and `syk`.
- **Nynorsk — `NbAiLab/normform-nynorsk-nasjonalbiblioteknorm` (12,487 reviewed choices, 19 `invalid_pairs`)**:
  - **Infinitives**: Consistent `-a` infinitive (`vera`, `bruka`, `gjera`, `halda`, `koma`, `setja`, `skriva`).
  - **Pronouns & function words**: `vi` (`me | vi`), `no` (`no | nå`), `difor` (`derfor | difor`), `blant` (`blant | millom`), `ikkje`.
  - **Nouns, adjectives & verbs**: `fyrst` / `fyrste`, `sundag` / `sundagen`, `ynskja` / `ynskjer`, `gje` / `gjev`, `vart` (`blei | vart`), definite feminine plural `-ene` (`bøkene`, `jentene`, `hendene`), and vowel alternation in strong/irregular verbs (`kjem`, `fer`, `held`, `tek`).

---

## 7. License & Credits

Licensed under the **Apache License 2.0** (see [`LICENSE`](LICENSE)).

Created by **Per Egil Kummervold** and **Magnus Breder Birkenes** at the **National Library of Norway (*Nasjonalbiblioteket*)**.
