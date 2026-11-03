# Provenance and validation record

## Export lineage

- `primary_dataset.csv`: unique `no_rag` items from `corrected_followup_results.csv`. The export retains `item_id`, `row_id`, `group_id`, `language`, `category`, `gold`, and `text`. Repeated records across models agreed on these fields during supplementary-package assembly.
- `diagnostic_subset_ids.csv`: distinct item IDs in `matched_corrected.csv`, checked against the primary export. All 384 IDs are present; they represent 48 source rows, 48 groups, and 64 language-specific items per category.
- `english_extension.csv`: English text and proposed labels from `extension_human_review.csv`, followed by explicit author approval of all 60 displayed labels without changes. The exported `gold` field records those accepted labels. It does not silently replace the earlier pending-review record.

The source files were supplied in `corrected_audit_20261009_134844.zip`. This compact repository does not include that entire working archive.

## Validation scope

The authors report manual creation and validation of both datasets and proficiency in the covered languages. Later approval explicitly accepts all displayed English extension labels unchanged. Individual reviewer identities, actual review dates, and independent annotation histories are not supplied. No independent external review or agreement coefficient is asserted.

## Unresolved historical provenance

The historical candidate protocol contains the construction description `Assistant-authored controlled counterfactual candidates` and marks human validation as incomplete at that stage. The later author statement reports manual author creation and completed author validation. Later label approval updates the acceptance status but does not establish which construction description is correct. Both accounts are retained here without inferring an undocumented creation process. Author reconciliation is required before this issue can be marked resolved.

No earlier timestamp has been repurposed as the actual human-review date. No translations have been added to the English-only extension.

## Source fingerprints

These fingerprints identify the exact archived source files; they are separate from the checksums of the compact exports.

| Source | SHA-256 |
| --- | --- |
| `corrected_followup_results.csv` | `c68e252d002dca6f94c418d2ba38eee7200317b55179c018c91749c8fc455371` |
| `matched_corrected.csv` | `2f7b8419f5cac6ff50d8e89fd8bdcd85590dbd6325f13e92998e125a75408a06` |
| `extension_human_review.csv` | `4ff4c7b730d3fb3626b332db95ff851e1521aedd1738acb515604f41aa5b12fc` |
| `candidate_protocol.json` | `92a1f6e77c14f3a6f72a2235d7af78fdda0206158b3c1e0037402da89a2a333c` |
