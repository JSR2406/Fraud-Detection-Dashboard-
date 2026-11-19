# Engineering Audit — Fraud Detection Dashboard

Audited the existing application at `c0a825520d88be78f61f060d9c5d2809857fd9b3`.

## Findings

- Preprocessing ran before train/test splitting, leaking imputation and category vocabularies across partitions.
- The trainer passed the holdout test set to LightGBM early stopping and used it for model evaluation.
- The split was random rather than chronological despite `TransactionDT` being available.
- Prediction silently truncated mismatched features and used a fixed 0.5 cutoff.
- `LabelEncoder` could not reliably handle unseen inference categories.
- Missing/corrupt real data was silently replaced by synthetic data, whose metrics could be mistaken for real performance.
- The per-row feature-contribution chart was an importance-based approximation, not a model explanation.
- Models and fitted preprocessing lived only in Streamlit session state.

## Remediation plan

The production path should use explicit `real`/`demo` loading, propagate real data errors, validate IDs/amount/target, split raw data chronologically into train/validation/test, fit a complete preprocessing/model pipeline on train only, use validation for tuning and threshold selection, and reserve the final test partition for final evaluation. Persist trusted model bundles with feature schema and threshold. Use safe unknown-category handling and reject schema mismatches. SHAP attributions must be labeled model explanations rather than causal effects.

## Caveats

IEEE-CIS `TransactionDT` is relative elapsed time, not a calendar timestamp. Do not infer calendar semantics from derived time features. Synthetic data is only suitable for UI and pipeline smoke tests. Business-cost values are assumptions, not measured savings. Joblib artifacts are executable serialization and must only be loaded from trusted locations.
