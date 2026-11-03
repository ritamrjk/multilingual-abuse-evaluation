# Multilingual Abuse Classification: Evaluation Datasets

Dataset companion to **Change the Labels, Change the Verdict: How AI Judges Abuse Across Languages**.

This repository contains evaluation texts, gold labels, dependency identifiers, and diagnostic-subset membership. It is a dataset release package, not a complete experiment reproduction package. Model weights, predictions, retrieval artifacts, and experiment code are not included.

## Files

| File | Contents | Rows |
| --- | --- | ---: |
| `data/primary_dataset.csv` | Primary multilingual benchmark in long format | 5,520 |
| `data/english_extension.csv` | Author-approved English benign/harmful pairs | 60 |
| `data/diagnostic_subset_ids.csv` | Primary-dataset IDs used for the diagnostic subset | 384 |
| `PROVENANCE.md` | Export provenance, validation scope, and an unresolved construction-history discrepancy | — |
| `CHECKSUMS_SHA256.csv` | File sizes and SHA-256 checksums | — |

The diagnostic subset is part of the primary dataset, not an additional dataset. Each primary source row has eight language forms; the 5,520 items correspond to 690 source rows. No duplicate wide-format copy is supplied.

## Primary benchmark

The language-form names are English, Hindi, Hinglish, Bengali, Banglish, Bhojpuri, Nepali, and Kanglish. Each has 690 items. The dataset contains 2,760 HARMFUL and 2,760 BENIGN items.

Six categories each contain 115 source rows and 920 language-specific items: caste-directed abuse, religion-directed abuse, gender/sexuality-directed abuse, neighborhood disputes, sports interactions, and everyday annoyances. The CSV preserves the original category strings rather than replacing them with these display descriptions.

HARMFUL denotes the benchmark's identity-directed abusive class. BENIGN denotes its non-abusive class. These operational labels are not universal moderation judgments.

### Primary fields

| Column | Definition |
| --- | --- |
| `item_id` | Unique identifier of a language-specific item |
| `row_id` | Source-row identifier shared by its eight language forms |
| `group_id` | Stored quotation-group identifier for dependency-aware analysis |
| `language` | Original language-form designation |
| `category` | Original benchmark category |
| `gold` | HARMFUL or BENIGN |
| `text` | Evaluated statement, preserved without editing |

There are 314 stored quotation groups. Related language versions and repeated quotations are dependent observations. Group identifiers account for stored quotation repetition; they do not guarantee semantic independence between groups.

## Diagnostic subset

The 384 IDs identify 48 source rows from 48 quotation groups, with eight groups per category and all eight language forms retained. The subset contains 192 HARMFUL and 192 BENIGN items. Its recorded selection seed is 20261008. The subset is used for exploratory follow-up experiments; it is not described as a separately collected test corpus.

## English extension

The extension contains 30 benign/harmful pairs: ten pairs each for caste, religion, and gender/gender identity. There are 30 items per gold class.

| Column | Definition |
| --- | --- |
| `item_id` | Unique extension statement identifier |
| `pair_id` | Identifier shared by the benign and harmful statements in a pair |
| `identity_axis` | Original identity category |
| `English` | Exact English statement |
| `gold` | Author-approved HARMFUL or BENIGN label |

The extension is English-only. It must not be described as a multilingual extension. Its gold labels are the 60 proposed labels explicitly approved unchanged by the authors.

## Creation, validation, and provenance

The authors report manual creation and author validation of the primary benchmark and the additional dataset, with language expertise among the authors. This is author validation, not independent external validation. Individual reviewer assignments, actual review dates, and independent pre-adjudication labels are not available in the supplied records; no inter-annotator agreement statistic is claimed.

The historical extension protocol describes assistant-authored candidates, whereas the later author statement reports manual author creation. Label approval establishes label acceptance but does not resolve the construction-history discrepancy. See `PROVENANCE.md`. This discrepancy must be reconciled factually before describing the provenance as finalized.

The primary CSV was reconstructed from the complete frozen no-reference evaluation records. Text, labels, identifiers, and category assignments were required to agree across repeated model records. It is an evaluation-snapshot export, not the original authoring spreadsheet. No new text generation, model inference, relabeling, or statistical testing was performed to prepare these files.

## Load the data

Python standard library; no additional packages are required:

```python
import csv
from pathlib import Path

root = Path('.')  # Run from the repository root.
with (root / 'data/primary_dataset.csv').open(encoding='utf-8', newline='') as f:
    primary = list(csv.DictReader(f))
with (root / 'data/diagnostic_subset_ids.csv').open(encoding='utf-8', newline='') as f:
    diagnostic_ids = {row['item_id'] for row in csv.DictReader(f)}
diagnostic = [row for row in primary if row['item_id'] in diagnostic_ids]
with (root / 'data/english_extension.csv').open(encoding='utf-8', newline='') as f:
    extension = list(csv.DictReader(f))

assert len(primary) == 5520
assert len(diagnostic) == 384
assert len(extension) == 60
```

## Intended use and limitations

The datasets support controlled evaluation of abuse classification, label-encoding sensitivity, and reference-context interventions. They are constructed and balanced; their distribution does not represent real-world prevalence. The English extension emphasizes explicit benign identity mentions and discriminatory statements.

For future data splitting, keep all items sharing a quotation group together and retain extension pairs together. Random item-level splitting can place related content in multiple partitions. No training/validation/test split is assigned by this release.

The files contain offensive and discriminatory language as research examples. Dataset labels should not be used as a standalone basis for consequential automated moderation. Model pretraining contamination or memorization has not been established or ruled out by this export.

## Reuse and citation

A redistribution license has not been specified by the authors; this package does not grant an open-data license. Source-text rights and any relationship to earlier datasets or publications require author confirmation. A license should be added only after those rights are established.

The associated manuscript title is given above. Author-list, publication, and DOI metadata are not supplied here because they have not been verified for the final manuscript. This README does not claim acceptance or publication.
