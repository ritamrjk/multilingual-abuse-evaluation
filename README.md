# Multilingual Abuse Classification

## Current matched-pair study

The study evaluates 36 benign–harmful scenario pairs across eight language forms,
yielding 576 evaluated items. Eleven instruction-tuned model checkpoints produced
95,040 responses across three label encodings and five reference conditions.

### Data and downloads

- [Evaluated benchmark](matched-pair-study/evaluated_benchmark_long.csv)
- [Analysis code, model revisions, reference wording, and audit](matched-pair-study/)
- [Online Resource 1 — Extended results](https://github.com/ritamrjk/multilingual-abuse-evaluation/releases/download/matched-pair-v1/Online_Resource_1_Extended_Results.zip)
- [Reproducibility archive — Saved responses and analysis code](https://github.com/ritamrjk/multilingual-abuse-evaluation/releases/download/matched-pair-v1/final_multilingual_11.zip)

Online Resource 1 contains the extended results, class recall and response-type
breakdowns, paired comparisons, and the analysis audit.

### Recompute the reported results

Extract `final_multilingual_11.zip` and open a terminal in the extracted folder.
With Python, NumPy, pandas, and Matplotlib installed, run:

    python final_cpu_analysis.py --inputs raw --out regenerated

The script reads the saved model responses and writes the regenerated results
to `regenerated/`. This command reproduces the analysis stage using saved outputs.

## Archival datasets

The repository also preserves earlier evaluation materials.

| File | Contents |
| --- | --- |
| [`data/primary_dataset.csv`](data/primary_dataset.csv) | Archival multilingual benchmark with 5,520 language-specific items derived from 690 source rows |
| [`data/english_extension.csv`](data/english_extension.csv) | English-only extension with 60 statements organised into 30 benign–harmful pairs |
| [`data/diagnostic_subset_ids.csv`](data/diagnostic_subset_ids.csv) | Identifiers for 384 items selected from the archival benchmark |
| [`tables/`](tables/) | Supporting tables for the archival evaluation |
| [`PROVENANCE.md`](PROVENANCE.md) | Export history, validation scope, and provenance notes for the archival materials |

The datasets contain examples of offensive and discriminatory language for
research. The labels describe the operational definitions used in these
evaluations. The constructed class balance does not estimate real-world prevalence.
