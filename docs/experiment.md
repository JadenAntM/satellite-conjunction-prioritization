# Experiment protocol

## Main question and contrast

The primary contrast is **Random Forest D minus Random Forest B at the three-day cutoff**: does visible history add ranking value beyond a full latest-message snapshot? Raw latest-risk ranking establishes the simple ranking reference; A Logistic establishes a learned one-field comparator. B/C/D across Logistic Regression, Random Forest and Histogram Gradient Boosting characterize information/model dependence. All fixed candidates are reported, including weaker ones.

Three days is primary. Two and four days are matched sensitivities using identical event IDs, endpoint labels, related groups, outer folds and training IDs. Model and horizon choices are not repeatedly optimized using reported evaluation scores.

## Information sets

- A: latest visible log-risk.
- B: 97 explicit safe latest-message fields.
- C: sequence count/span and six causal statistics for risk, miss distance and radial/along-track/cross-track target/chaser position sigmas: mean, population SD, maximum, first-to-latest change, change per visible day and last change (50 fields).
- D: B + C, 147 fields.

C includes level information and can equal the latest value for a one-message visible sequence. It is not pure evolution. D−B isolates incremental history for the chosen estimators. A null result does not prove histories contain no useful information for every possible model.

## Splitting and leakage prevention

Five-fold StratifiedGroupKFold, seed 20261005, splits conservative related groups rather than individual messages. Preprocessing fits only each training partition. Absolute dates are anonymized, so the protocol does not claim temporal generalization.

The optional leave-one-mission-out run evaluates A Logistic and B/D Random Forest at every horizon. Related groups are required to belong to one mission. No training group may overlap the held-out mission. Mission sizes and positive counts are unequal; zero-positive held-out missions have undefined AP/recall. Pooled mission results do not represent a verified daily review queue.

## Fixed models

| Family | Settings |
| --- | --- |
| Logistic Regression | C=1, max_iter=3000, seed=42; training standardization |
| Random Forest | 200 trees, depth≤8, leaf≥10, two workers, seed=42 |
| Histogram Gradient Boosting | 150 iterations, learning rate=.05, leaves≤15, L2=1, no early stopping, seed=42 |

Numeric fields receive training median imputation with missing indicators. Latest log-risk and `history__risk__*` retain their scale; other numeric fields receive signed log1p to soften tails. The object type receives training-fitted category encoding, with unseen categories ignored. IDs are never categorical features.

## Class imbalance

The matched population contains 160 positives out of 8,694 events (1.84%). The fixed models are unweighted and do not synthesize or oversample events. Handling imbalance here means appropriate grouping, supported positive counts, ranking-oriented evaluation and explicit misses/precision, rather than maximizing accuracy or forcing balanced labels.

If weighting/sampling is later investigated, apply it only inside training folds, keep evaluation prevalence unchanged, record it as a different experiment, and assess its effect on calibration. Existing evaluation scores must not become an unacknowledged tuning set.

## Metrics and capacity

Average Precision is primary. Report ROC-AUC, Brier/log loss for binary-endpoint probabilities, and analyst-budget metrics at 1/2/5/10%: reviews, high endpoints captured/missed, final-low reviews, precision, recall, prevalence lift and incremental capture over latest risk. Raw log-risk receives ranking metrics but no Brier/log loss.

Capacity is ceil(N × budget). Boundary ties receive their expected label-blind allocation, avoiding label-based sorting. Minimum review fractions for 80/90/95% capture include whole boundary ties. These are retrospective workload curves and cannot be used as evaluation-label-fitted deployment thresholds. A final-low review is not proof of unnecessary work.

For D−B and RF B/D versus latest ranking, report 1,000 paired related-group and 1,000 paired whole-mission bootstrap resamples of saved OOF scores. Reuse the same resamples for each pair. These conditional intervals exclude fitting uncertainty and have no multiplicity adjustment. Report fold results as well as pooled metrics. Mission resampling and model fitting with mission holdout are distinct analyses.

## Optional calibration

At the primary horizon, each outer training fold supplies three inner grouped folds (seed 20261035 + outer fold). Calibrators fit only inner OOF scores and training labels. Platt uses Logistic Regression C=1 on clipped score logits. Isotonic requires at least 50 examples of each class in the outer training set. The raw refit must reproduce the original saved outer scores.

Evaluate raw/Platt/isotonic on outer labels without selecting a method and rescoring those same labels as confirmation. Save training IDs/inner folds and outer provenance. Fixed reliability bins retain counts and descriptive Wilson intervals; they assume independent Bernoulli observations and are imperfect under mission dependence. Tail support can be small even when global Brier improves. These probabilities describe later-message endpoints, not physical collision.

## Reproducibility and claims

Run manifests record settings, input/source hashes, package versions and output hashes. Input tables and run names are protected from overwriting. A renamed or reseeded run is not independent validation of this previously explored dataset.

The project can assess retrospective early prioritization and the value of information. It cannot establish actual collisions, avoided manoeuvres, fuel savings, satellite identities, dismissible alerts, future-date performance or a reliably calibrated physical hazard probability. Budget gains need not hold at every capacity; history benefit must be supported by the D−B comparison rather than assumed from feature importance.
