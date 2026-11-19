# Engineering Audit — Fraud Detection Dashboard

Baseline audited at `c0a825520d88be78f61f060d9c5d2809857fd9b3` (pre-upgrade `main`).

## Findings and exact fixes

| # | Baseline finding | Fix implemented | Verification |
|---|---|---|---|
| 1 | `src/preprocessing.py` fit imputers/encoders on the full frame before splitting | New `fraud_ml/data/splitting.py` partitions raw data chronologically; `fraud_ml/features/pipeline.py` fits imputation/scaling/one-hot only on the training partition inside a `Pipeline` | `test_chronological_split_orders_data_and_rejects_invalid_partitions`, `test_preprocessor_fits_and_transforms_unseen_category` |
| 2 | `src/model_trainer.py` passed the holdout test set to LightGBM early stopping | `fraud_ml/models/train.py` uses validation for all selection; the final test partition is scored once after threshold/calibration are fixed. `fraud_ml/models/tuning.py` optimizes validation PR-AUC only | `test_train_persist_reload_predict_end_to_end`, `test_training_bundle_includes_calibration_and_metadata`; `metrics.json` records `final_test_used_once: true` |
| 3 | Random `train_test_split` ignored `TransactionDT` | Chronological train/validation/test split with explicit fractions and both-class checks | `test_chronological_split_orders_data_and_rejects_invalid_partitions` |
| 4 | Unknown categoricals broke inference | `OneHotEncoder(handle_unknown="ignore")` plus `unknown` sentinel in row-local engineering; persisted feature schema is checked strictly | `test_feature_engineering_keeps_schema_and_handles_unseen_values`, `test_predictor_rejects_bad_schema_and_invalid_amount` |
| 5 | Prediction silently truncated mismatched features | `FraudPredictor` rejects unexpected fields and asserts exact feature-column order | `test_predictor_rejects_bad_schema_and_invalid_amount` |
| 6 | Fixed 0.5 decision threshold | Validation-tuned `optimize_threshold` (max F1 / min precision / review capacity / business cost) stored in the artifact | `test_metrics_and_threshold_selection_are_consistent`, `test_threshold_is_never_degenerate` |
| 7 | Synthetic fallback could be mistaken for real performance | Explicit `real`/`demo` modes; real mode raises `FileNotFoundError`; demo label contains "synthetic"; dashboard warning banner | `test_demo_mode_is_explicit_reproducible_and_labeled`, `test_real_mode_fails_clearly_instead_of_falling_back` |
| 8 | Approximate contribution chart presented as explanation | `fraud_ml/models/explain.py` provides bounded SHAP values labelled as model (not causal) explanations, with graceful degradation | `test_explanations_degrade_gracefully_when_shap_unavailable` |
| 9 | Models existed only in Streamlit session state | Joblib bundle persisted with pipeline, calibrator, threshold, feature schema, dataset fingerprint and code revision; strict loader | `test_train_persist_reload_predict_end_to_end`, `test_training_bundle_includes_calibration_and_metadata` |
| 10 | Uncalibrated scores treated as probabilities | Validation-only score calibration (isotonic/Platt) with Brier before/after and calibration curve in the artifact | `test_score_calibration_reports_brier_and_transforms` |
| 11 | No business-cost or review-capacity evaluation | `fraud_ml/models/business.py` with explicit, documented assumptions, amount-at-risk and fraud capture rate | `test_business_evaluation_uses_explicit_assumptions` |
| 12 | No experiment tracking, registry gates or monitoring | MLflow tracking with local JSON fallback; gate-based registry with aliases and rollback; dataset quality/drift reports | `test_registry_gates_promotion_and_rollback`, `test_monitoring_profile_and_drift_report` |

## Additional hardening

- Threshold candidates are strictly inside `(0, 1)`; degenerate 0/1 cutoffs can no longer be persisted and then rejected at inference.
- Business costs are explicitly labelled estimates from assumptions, never realized savings.
- SHAP/native-extension failures are caught so the dashboard and API keep working.
- Joblib artifacts are treated as executable code and loaded only from trusted operator-provisioned paths.

## Caveats

- IEEE-CIS `TransactionDT` is relative elapsed time, not a calendar timestamp; derived hour/day features carry no calendar semantics.
- Synthetic data is only suitable for UI and pipeline smoke tests.
- Optuna, MLflow and Pandera are optional: graceful fallbacks are used when they are absent, and fallbacks are labelled as such.
- No real IEEE-CIS dataset is bundled, so no real-world performance claim is made.
