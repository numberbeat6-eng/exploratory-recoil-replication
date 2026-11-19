# Replication materials for You Recommend, I Retreat

These materials accompany *You Recommend, I Retreat: Defensive Retrenchment and Content Selection in Exploratory Recommendation*.

## Data

Download and extract the original archives from the data providers. Pass the directory containing the CSV files to the runner.

| Study | Download | Data files |
|---|---|---|
| Study 1 | [KuaiRand-1K](https://zenodo.org/records/10439422), `KuaiRand-1K.tar.gz` | `log_standard_4_08_to_4_21_1k.csv`, `log_standard_4_22_to_5_08_1k.csv`, `log_random_4_22_to_5_08_1k.csv`, `video_features_basic_1k.csv` |
| Study 2 | [KuaiRec 2.0](https://zenodo.org/records/18164998), `KuaiRec.zip` | `small_matrix.csv`, `item_categories.csv` |

The dataset papers describe how [KuaiRand](https://doi.org/10.1145/3511808.3557624) and [KuaiRec](https://doi.org/10.1145/3511808.3557220) were collected.

## Run

Use Python 3.13 in a dedicated environment. Run these commands from the extracted package directory.

```text
python -m pip install -r requirements.txt
python run.py --kuairand "path/to/KuaiRand-1K/data" --kuairec "path/to/KuaiRec/data"
```

To run one study, pass just its data argument. The runner writes results to `output/study1` and `output/study2` by default. Use `--output` to choose a different output directory. Allow time and disk space to read and process the full logs, especially the KuaiRand files.

## Find and check the results

Study 1 uses 21,756 user-days in the main daily models. The primary event sample has 12,172 insertions and the common sample has 6,192. The estimated daily coefficient for unfamiliar effective play is approximately −0.007288.

| Manuscript result | Study 1 output |
|---|---|
| Tables 1–4 | `table1_descriptive.csv`, `table2_main.csv`, `table3_events.csv`, `table4_heterogeneity.csv` |
| Tables A1 and A2 | `tableA1_robustness.csv`, `tableA2_insertion_rate.csv` |
| Figures 3–5 | `figures/parallel_trends.pdf`, `figures/event_window_attenuation.pdf`, `figures/baseline_heterogeneity.pdf` |
| Figures A1–A6 | In `figures/`, the PDFs named `sample_construction`, `exposure_sequence`, `preperiod_quartiles`, `causal_forest_distribution`, `shap_importance`, and `shap_effective_play` |
| Baseline acceptance of inserted videos | `s1_insertion_quality.json` |

`Study1/table4_reported.csv` records the aggregates as printed in Table 4. Re-estimating the forest produces small numerical differences from the original fit. For example, the unfamiliar effective-play Q4–Q1 estimate was −0.002249 in the verification run and −0.002239 in the manuscript.

Study 2 uses a matrix with 1,411 users and 3,327 videos. The familiar and unfamiliar completion means are approximately 43.0% and 29.8%. The corrected completion-based phase analysis uses 5,639 adjacent pairs and yields slopes of approximately 0.320 and 0.248 without and with the current familiar completion control. The seven-second analysis uses each viewing record's durations.

| Manuscript result | Study 2 output |
|---|---|
| Table 5 and Figure 6 | `table_5.json`, `Figure_6.pdf` |
| Section 4.2 and Table B1 | `five_video_batch.json`, `table_B1.csv` |
| Section 4.3 | `temporal_completion.json` |
| Table 6 and Figure 7 | `table_6.csv`, `Figure_7.pdf` |
| Table B2 and the seven-second phase check | `table_B2.csv`, `temporal_effective_play.json` |
