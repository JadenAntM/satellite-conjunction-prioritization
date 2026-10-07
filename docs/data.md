# Data contract

## Official source

- Dataset: **Collision Avoidance Challenge — Dataset**, ESA Space Debris Office.
- Record: [Zenodo 4463683](https://zenodo.org/records/4463683).
- Original challenge: [ESA Kelvins](https://kelvins.esa.int/collision-avoidance-challenge/data/).
- Dataset licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Release: anonymized CDMs from 2015–2019; 199,082 messages, 15,321 events, 103 columns.
- Archive bytes: 221,128,642.
- SHA-256: `df1500146705305006ea506eeccc97f5c2e9593928d6650c503c4f5b5529d0bb`.
- Published MD5: `d19dc8875229f2f6893253c38adddc87`.

The downloader uses the official record's file endpoint. Parsing verifies the pinned SHA-256 and reads the raw gzip member directly from the ZIP. No other competition split or private-label table is required. Data are not committed to this repository. The explicit schema and feature allowlist live in `src/conjunctions/schema.py`.

## Unit of observation and target

One model row represents an entire conjunction event at a specified cutoff. `time_to_tca` is in days; larger values are earlier messages. At cutoff k, predictors use only messages with `time_to_tca >= k`, including the latest of those messages.

For the label, take the last message in the complete event sequence and require `0 <= time_to_tca < 1`. Positive means that message's base-10 log-risk is at least −6. Events ending after TCA, ending at least one day before TCA, lacking visible messages, or containing fewer than two messages are excluded. A post-TCA ending is excluded even if an earlier pre-TCA message exists; this preserves the explicit endpoint contract.

The endpoint is **high risk in the final qualifying message**, not a physical collision and not “ever crosses the threshold.” Classifier probabilities and the message's physical risk estimate have different meanings. `final_risk`, labels, IDs and fold/group metadata never enter model features.

## Cleaning and grouping

Exact duplicate rows are removed. Entire events with invalid risk/time, conflicting messages at the same time, or inconsistent mission IDs are quarantined. Nonfinite numeric feature values become missing. Invalid event/mission identifiers are rejected. Missing predictors are retained and imputed in training folds; outcomes are not imputed.

Conservative related-event groups join events sharing mission, rounded message time, risk and miss-distance signatures. Geometry may differ between members. This proxy helps contain reciprocal/related encounters but is not a certified physical identity join. Hashing/rounding and thresholds are fixed in code, and no satellite identities are inferred.

## Main population

| Cutoff | Native eligible events | High endpoints |
| --- | --- | --- |
| 2 days | 10,032 | 201 |
| 3 days | 9,388 | 179 |
| 4 days | 8,694 | 160 |

The main experiment uses the four-day-eligible event IDs at all horizons: 8,694 rows, 160 positives, 8,637 related groups, 19 missions. This selects events with sufficiently early usable histories. The preparation audit records these measured counts; it is generated locally alongside the event tables.

## Field semantics and missingness

Risk is already in log10 units. Miss distance and position sigmas are metres; velocity and velocity sigmas are m/s. Orbit/observation and covariance-related attributes describe the current message. Missingness varies by field and object class. Median imputation, missing indicators, numerical transforms and category encoding are fitted on training rows only.

`c_object_type` identifies the chaser class: DEBRIS, PAYLOAD, ROCKET BODY, UNKNOWN or literal TBA. PAYLOAD does not establish active/inactive status. UNKNOWN events are retained; neither physical identities nor activity classes are reconstructed from anonymized orbit/mission fields. Source IDs and F10/F3M/AP/SSN are excluded from predictors.

Usable absolute dates, operator actions and collision outcomes are not provided. The benchmark has already been explored, so internal folds cannot be called a fresh independent holdout. A genuinely new validation source would require its own source/timing/label audit.
