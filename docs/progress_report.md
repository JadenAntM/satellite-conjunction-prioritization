# Progress report preparation

Group 69 studies early satellite conjunction prioritization: whether the latest full CDM improves over latest risk, and whether visible history adds value beyond the snapshot. Three days is primary. The target is a high-risk final qualifying CDM, not an observed collision. The satellite project is the current team direction.

## Generate evidence without retraining

From the repository root after completing the main run:

```bash
python -m conjunctions milestone --run results/runs/main --output results/milestone/main
```

Open `results/milestone/main/REPORT_GUIDE.md` for tables, caveats and file descriptions. Open `ranking_demo.html` in any browser, including offline. The command verifies the completed run's prediction hash, recomputes point estimates, records provenance, and refuses existing output directories. It does not modify the run, refit models, select a winning model or regenerate bootstrap intervals. Choose a new output directory for subsequent exports. It also works with completed mission runs, but those have different evaluation meaning and should be exported separately.

Use `primary_results.tex` as a starting point for the supplied course template. It is a wide table: use a two-column-spanning `table*`, or split columns into ranking and probability tables. Do not squeeze every available metric into the main narrative. The PNG figure can be included directly with `graphicx`; SVG is available for editing or external export. All generated artifacts stay ignored by Git; include selected outputs in the A2L archive.

## Course requirements and writing plan

The supplied Fall 2026 instructions state November 10 as the due date; they do not specify a submission time. Check A2L for the cutoff time. Use the supplied LaTeX template with around four pages of written content, allowing additional space for figures/tables. Preserve its formatting. These requirements come from the course instructions; the evidence choices below are team preparation recommendations.

| Item | Weight | Evidence and explanation |
| --- | --- | --- |
| Dataset and preprocessing | 15% | `docs/data.md`, preparation audit: source/license, raw vs event counts, matched cohort, labels, exclusion rules, cleaning and fold-fitted imputation. No manual annotation is used. |
| Introduction and related work | 10% | Adapt the proposal's motivation; cite at least five relevant references in the report. The proposal has four. Explain relationships to prior work rather than listing citations. |
| Inputs and features | 10% | Explain A/B/C/D and dimensions, causal history statistics, categorical encoding, exclusions and selection. C contains levels as well as trends. |
| Model implementation | 20% | `src/conjunctions/models.py`, config and run manifest: describe learning objective/split criterion, optimization/training procedure and important settings for models actually run. Include a simple baseline and latest-risk ranking. |
| Evaluation and preliminary results | 20% | `docs/experiment.md`, verified exported tables, original paired intervals and fold metrics: explain grouped CV, matched cutoffs, AP, budget precision/recall, uncertainty and limits. |
| TA feedback, reflection and plans | 20% | Record actual feedback and responses, what changed since the proposal, findings that affect next steps, and feasible remaining tasks. No explicit TA approval is required. |
| Team contributions | 5% | Accurate completed contributions from Jaden Moore, Varun Pathak and Lukhsaan Elankumaran; confirm with each person. Do not infer contributions from this checklist. |

Follow the template's Introduction, Related Work, Dataset, Features, Implementation, Results and Evaluation, Feedback and Plans, and Team Contributions sections. Remove instructional examples and placeholder bibliography entries.

## Focused error analysis

1. Compare latest-risk, B Random Forest and D Random Forest using the same primary-cutoff events.
2. Read `error_summary.csv`. At 5% capacity, inspect high endpoints missed/partially missed and low endpoints reviewed/partially reviewed. Inclusion weights share tied capacity; do not force tied events into an arbitrary binary queue.
3. Inspect `case_studies.csv` alongside `event_analysis.csv`. Examples are deliberately selected by largest final-minus-latest log-risk increase, then event ID. Categories overlap; do not treat them as representative samples or add their counts.
4. Use `history_rank_changes.csv` to identify events moved earlier and later by history. Positive gain means an earlier average rank. Check labels before interpreting movement as useful. Rank shifts alone do not establish a general performance gain or a causal explanation.
5. Use the offline demo to explain these comparisons to teammates. Outcomes are hidden initially but embedded in the HTML; filters based on outcomes are explicitly retrospective. The demo uses saved out-of-fold scores and is not a live operational predictor.
6. Report null or mixed history results. Use existing paired confidence intervals before claiming benefits. Bootstrap intervals condition on fitted scores and do not include model-fitting uncertainty. Do not retune repeatedly on these evaluation labels and call the new results independent validation.

## Reproducibility handoff

One teammate should try the following in a fresh clone and virtual environment, following README setup. This is a checklist, not a claim that an independent machine has passed it.

```bash
python -m pytest -q
python -m conjunctions download
python -m conjunctions prepare
python -m conjunctions run --name progress_check
python -m conjunctions milestone --run results/runs/progress_check --output results/milestone/progress_check
```

Record commit SHA (`git rev-parse HEAD`), Python version, commands, test outcome and run manifest. Verify population counts against the data contract and ensure any differences are explained. Existing processed data and run names cannot be overwritten: use a fresh clone or new directories/names. Download requires internet and training takes longer than exporting saved results. A rerun supports reproducibility; it is not new independent evidence or automatically new scientific progress.

## Submission checklist

- Complete report PDF in the provided template; at least five cited references; remove placeholders.
- Include clear preliminary results and limitations, actual TA feedback, next steps and contributions.
- Submit report, code and data information together as an archive to A2L. A GitHub URL alone does not replace the requested code submission.
- Because the ESA source archive exceeds 50 MB, include its source link and download instructions rather than the full dataset. Include attribution/license information. Follow A2L instructions if they add requirements.
- Include source/config/tests/docs, dependency specifications, selected tables/figures and relevant manifests. Exclude `.venv`, caches, `.git`, secrets, downloaded archives and large intermediate tables.
- Verify that the archive's README explains setup and reproduction; check the archive contents before uploading.
- Keep claims to retrospective final-message prioritization. No verified collision prediction, automatic alert dismissal, avoided manoeuvres or fuel-saving claim follows from these results.
