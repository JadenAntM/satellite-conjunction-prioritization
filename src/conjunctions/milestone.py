"""Read-only analysis of completed out-of-fold predictions for the course milestone."""

import html
import json
import math
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from .archive import sha256
from .metrics import budget_metrics, score_metrics, validate_scores


FOCUS = ["latest_risk", "B_random_forest", "D_random_forest"]
NOTICE = (
    "Exploratory internal validation on a previously explored benchmark. "
    "Outcomes are final-message endpoints, not observed collisions. "
    "Review budgets describe this pooled retrospective cohort, not a daily operational queue."
)


def review_weights(scores, fraction):
    """Expected label-blind inclusion at capacity, sharing boundary ties equally."""
    scores = np.asarray(scores, dtype=float)
    if not len(scores) or not np.isfinite(scores).all() or not 0 < fraction <= 1:
        raise ValueError("Expected finite scores and a review fraction in (0, 1]")
    capacity = math.ceil(len(scores) * fraction)
    boundary = np.partition(scores, len(scores) - capacity)[len(scores) - capacity]
    above, tied = scores > boundary, scores == boundary
    return above.astype(float) + tied * ((capacity - above.sum()) / tied.sum())


def load_predictions(run):
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    if "completed_utc" not in manifest:
        raise ValueError("Milestone export requires a completed run")
    if sha256(run / "predictions.csv") != manifest["prediction_sha256"]:
        raise ValueError("Saved predictions differ from the completed run")
    data = pd.read_csv(run / "predictions.csv")
    required = {"event_id", "cutoff", "fold", "mission_id", "evaluation_group", "label", "final_risk", "latest_risk"}
    if not required.issubset(data.columns) or data.empty:
        raise ValueError("Missing event metadata or predictions")
    if data[list(required)].isna().any().any() or data.duplicated(["cutoff", "event_id"]).any():
        raise ValueError("Missing metadata or duplicate event predictions")
    if not np.isfinite(data.final_risk).all() or not (data.label == (data.final_risk >= -6).astype(int)).all():
        raise ValueError("Labels differ from final-message endpoints")
    names = [c for c in data if c in ("latest_risk", "training_prevalence") or re.fullmatch(r"[ABCD]_(logistic|random_forest|hgb)", c)]
    for name in names:
        validate_scores(data.label, data[name], probability=name != "latest_risk")
    if set(data.cutoff.unique()) != set(manifest["config"]["cutoffs"]):
        raise ValueError("Prediction cutoffs differ from the run configuration")
    primary = manifest["config"]["primary_cutoff"]
    events = data[data.cutoff == primary].copy().sort_values("event_id").reset_index(drop=True)
    if events.empty:
        raise ValueError("Primary cutoff has no predictions")
    return manifest, data, events, names


def event_analysis(events, names):
    columns = [c for c in ["event_id", "evaluation_group", "mission_id", "fold", "label", "final_risk",
                           "latest__c_object_type", "latest_risk"] if c in events]
    result = events[columns].copy()
    result["retrospective_log_risk_increase"] = events.final_risk - events.latest_risk
    for name in names:
        result[name] = events[name]
        result[f"{name}__rank_min"] = events[name].rank(method="min", ascending=False).astype(int)
        result[f"{name}__rank_max"] = events[name].rank(method="max", ascending=False).astype(int)
        result[f"{name}__review_weight_5pct"] = review_weights(events[name], .05)
    if {"B_random_forest", "D_random_forest"}.issubset(names):
        # Positive means history moves the event earlier. Ties use average ranks.
        result["history_rank_gain"] = (
            events.B_random_forest.rank(method="average", ascending=False)
            - events.D_random_forest.rank(method="average", ascending=False)
        )
    return result


def write_latex(table, output, cutoff):
    """Plain LaTeX with no optional pandas styling/document dependencies."""
    lines = [r"\begin{table*}[t]", r"\centering", r"\begin{tabular}{lrrrrrr}", r"\hline",
             r"Approach & AP & ROC-AUC & Hits & Missed & Precision & Recall \\", r"\hline"]
    for row in table.itertuples():
        name = row.model.replace("_", r"\_")
        values = [row.ap, row.roc_auc, row.top5_hits, row.top5_missed, row.top5_precision, row.top5_recall]
        cells = ["--" if pd.isna(x) else f"{x:.3f}" for x in values]
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}",
              f"\\caption{{Exploratory {cutoff}-day out-of-fold results. Hits, misses, precision and recall use a 5\\% review budget; boundary ties share capacity.}}",
              r"\label{tab:primary-results}", r"\end{table*}"]
    (output / "primary_results.tex").write_text("\n".join(lines) + "\n")


def make_figures(events, names, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output.resolve() / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [n for n in FOCUS if n in names]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in selected:
        precision, recall, _ = precision_recall_curve(events.label, events[name])
        ap = score_metrics(events.label, events[name], name != "latest_risk")["ap"]
        axes[0].step(recall, precision, where="post", label=f"{name} (AP {ap:.3f})")
        fractions = np.linspace(.01, 1, 100)
        capture = [budget_metrics(events.label, events[name], f)["recall"] for f in fractions]
        axes[1].plot(fractions * 100, capture, label=name)
    axes[0].axhline(events.label.mean(), color="grey", linestyle="--", label="Cohort prevalence")
    axes[0].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="Cohort reviewed (%)", ylabel="High endpoints captured (recall)", xlim=(0, 100), ylim=(0, 1))
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.suptitle(f"{int(events.cutoff.iloc[0])}-day cutoff: retrospective out-of-fold ranking", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(output / f"ranking_curves.{ext}", dpi=180)
    plt.close(fig)


def write_demo(analysis, names, output, cutoff):
    selected = [n for n in FOCUS if n in names]
    columns = ["event_id", "mission_id", "label", "latest_risk", "final_risk", "retrospective_log_risk_increase"]
    for n in selected:
        columns += [n, f"{n}__rank_min", f"{n}__rank_max", f"{n}__review_weight_5pct"]
    columns = list(dict.fromkeys(columns))
    payload = analysis[columns].to_json(orient="records", double_precision=12).replace("<", "\\u003c")
    options = "".join(f'<option value="{html.escape(n)}">{html.escape(n)}</option>' for n in selected)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Satellite ranking demonstration</title><style>
body{font:16px system-ui;margin:32px auto;padding:0 20px;max-width:1300px;color:#182b3a;background:#f7f9fb}h1{font-size:27px}
label{display:inline-block;margin:8px 16px 8px 0}select,input,button{font:inherit;padding:5px}table{border-collapse:collapse;width:100%;background:white;font-size:14px}
th,td{text-align:left;border-bottom:1px solid #dde3e8;padding:9px}th{background:#e7eff5}p{line-height:1.5}.scroll{overflow:auto}.note{padding:14px;background:#e7eff5}.endpoint{background:#fff4d9}
</style><h1>Satellite close-approach rankings</h1>
<p>__CUTOFF__ days before closest approach. __NOTICE__</p>
<p class="note">These are held-out predictions from several folds. Scores from different models have different meanings:
latest risk is log10 message risk; learned scores estimate the probability of a high final-message endpoint.
Ranks are tie ranges. Event ID orders tied display rows only; evaluation shares review capacity across all tied events.</p>
<label>Rank by <select id="model">__OPTIONS__</select></label>
<label>Find event <input id="event" type="search" placeholder="Event ID"></label>
<label>Show <select id="filter"><option value="all">All endpoints</option><option value="high">High endpoints</option><option value="missed">High endpoints outside / partly outside top 5%</option><option value="low">Low endpoints inside / partly inside top 5%</option></select></label>
<label><input id="reveal" type="checkbox"> Reveal later outcomes for analysis</label>
<p id="summary"></p><div class="scroll"><table><thead id="head"></thead><tbody id="body"></tbody></table></div>
<p><button id="prev">Previous</button> <span id="page"></span> <button id="next">Next</button></p>
<p>Filters use later outcomes for retrospective analysis only. Rank and review inclusion always refer to the entire cohort, even when filtered.
No messages after the cutoff are model inputs. A low endpoint does not prove a review was unnecessary.</p>
<script type="application/json" id="data">__DATA__</script><script>
const rows=JSON.parse(document.getElementById('data').textContent), byId=id=>document.getElementById(id);let page=0;
function draw(){const m=byId('model').value,f=byId('filter').value,q=byId('event').value.trim(),reveal=byId('reveal').checked;
const w=r=>r[m+'__review_weight_5pct'];
const sorted=rows.filter(r=>(!q||String(r.event_id).includes(q))&&(f==='all'||f==='high'&&r.label===1||f==='missed'&&r.label===1&&w(r)<1||f==='low'&&r.label===0&&w(r)>0)).sort((a,b)=>b[m]-a[m]||a.event_id-b.event_id);
page=Math.max(0,Math.min(page,Math.ceil(sorted.length/50)-1));
const headers=['Rank (ties)','Event','Mission','Latest log-risk','Ranking score','Top 5% inclusion weight',...(reveal?['Later endpoint','Final log-risk','Log-risk increase']:[])];
byId('head').replaceChildren();const tr=document.createElement('tr');for(const h of headers){const th=document.createElement('th');th.textContent=h;tr.append(th)}byId('head').append(tr);
byId('body').replaceChildren();for(const r of sorted.slice(page*50,(page+1)*50)){const tr=document.createElement('tr');if(reveal&&r.label)tr.className='endpoint';
const lo=r[m+'__rank_min'],hi=r[m+'__rank_max'];const cells=[lo===hi?lo:lo+'–'+hi,r.event_id,r.mission_id,r.latest_risk.toFixed(3),r[m].toFixed(5),w(r).toFixed(3),...(reveal?[r.label?'High':'Low',r.final_risk.toFixed(3),r.retrospective_log_risk_increase.toFixed(3)]:[])];
for(const x of cells){const td=document.createElement('td');td.textContent=x;tr.append(td)}byId('body').append(tr)}
byId('summary').textContent=sorted.length+' matching events / '+rows.length+' total. Showing up to 50 per page.';
byId('page').textContent=sorted.length?'Page '+(page+1)+' of '+Math.ceil(sorted.length/50):'No matching events';byId('prev').disabled=page===0;byId('next').disabled=(page+1)*50>=sorted.length;}
for(const id of ['model','event','filter','reveal'])byId(id).addEventListener('input',()=>{page=0;draw()});
byId('prev').onclick=()=>{page--;draw()};byId('next').onclick=()=>{page++;draw()};draw();
</script></html>'''
    for key, value in {"__CUTOFF__": str(cutoff), "__NOTICE__": html.escape(NOTICE), "__OPTIONS__": options, "__DATA__": payload}.items():
        page = page.replace(key, value)
    (output / "ranking_demo.html").write_text(page)


def export_milestone(run, output):
    """Recompute point estimates, preserve the source run, and protect existing exports."""
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError("Milestone output already exists; choose a new directory")
    if output.resolve().is_relative_to(run.resolve()):
        raise ValueError("Keep milestone exports outside the source run directory")
    manifest, data, events, names = load_predictions(run)
    if events.label.nunique() != 2:
        raise ValueError("Primary cohort must contain both endpoint classes")
    metrics, budgets = [], []
    for cutoff, cohort in data.groupby("cutoff"):
        for name in names:
            metrics.append({"cutoff": cutoff, "model": name, **score_metrics(cohort.label, cohort[name], name != "latest_risk")})
            for fraction in (.01, .02, .05, .10):
                budgets.append({"cutoff": cutoff, "model": name, **budget_metrics(cohort.label, cohort[name], fraction)})
    metrics = pd.DataFrame(metrics)
    primary = manifest["config"]["primary_cutoff"]
    table = metrics[metrics.cutoff == primary][["model", "ap", "roc_auc", "brier", "log_loss", "top5_hits", "top5_missed", "top5_precision", "top5_recall"]]
    analysis = event_analysis(events, names)
    output.mkdir(parents=True)
    metrics.to_csv(output / "all_metrics.csv", index=False)
    pd.DataFrame(budgets).to_csv(output / "review_budgets.csv", index=False)
    table.to_csv(output / "primary_results.csv", index=False)
    write_latex(table, output, primary)
    analysis.to_csv(output / "event_analysis.csv", index=False)
    summary, cases = [], []
    for name in [n for n in FOCUS if n in names]:
        weight = analysis[f"{name}__review_weight_5pct"]
        for kind, mask in [("high_endpoint_missed", (analysis.label == 1) & (weight < 1)),
                           ("low_endpoint_reviewed", (analysis.label == 0) & (weight > 0)),
                           ("late_threshold_crossing", (analysis.label == 1) & (analysis.latest_risk < -6))]:
            subset = analysis[mask].copy()
            summary.append({"model": name, "category": kind, "events": len(subset),
                            "expected_reviewed": weight[mask].sum(), "expected_not_reviewed": (1 - weight[mask]).sum()})
            subset["analysis_model"], subset["category"] = name, kind
            subset = subset.sort_values(["retrospective_log_risk_increase", "event_id"], ascending=[False, True]).head(10)
            cases.append(subset)
    pd.DataFrame(summary).to_csv(output / "error_summary.csv", index=False)
    pd.concat(cases).to_csv(output / "case_studies.csv", index=False)
    if "history_rank_gain" in analysis:
        changed = analysis.assign(absolute_rank_change=analysis.history_rank_gain.abs())
        changed.sort_values(["absolute_rank_change", "event_id"], ascending=[False, True]).head(30).to_csv(output / "history_rank_changes.csv", index=False)
    make_figures(events, names, output)
    write_demo(analysis, names, output, primary)
    rows = ["| Approach | AP | Hits at 5% | Precision at 5% | Recall at 5% |", "|---|---:|---:|---:|---:|"]
    for row in table.itertuples():
        rows.append(f"| {row.model} | {row.ap:.3f} | {row.top5_hits:.2f} | {row.top5_precision:.3f} | {row.top5_recall:.3f} |")
    majority_accuracy = 1 - events.label.mean()
    text = f"""# Progress report evidence

{NOTICE}

Primary cutoff: {primary} days. Evaluation scheme: {manifest['scheme']}.
Cohort: {len(events):,} events; {int(events.label.sum()):,} high endpoints ({events.label.mean():.2%}).
Always predicting low gives {majority_accuracy:.2%} accuracy and zero high-endpoint recall.
This hard-label illustration is separate from the training-prevalence probability baseline.

{chr(10).join(rows)}

## Use these materials

- `primary_results.csv` / `.tex`: all available approaches at the primary cutoff. AP is not accuracy. The LaTeX table uses `table*` for the two-column course template; the CSV additionally includes Brier and log loss for a separate probability discussion.
- `all_metrics.csv`, `review_budgets.csv`: all horizons, including weak candidates. Hits can be fractional because boundary ties share capacity.
- `ranking_curves.png` / `.svg`: primary-cutoff precision-recall and retrospective capture versus workload. This exporter does not recompute uncertainty intervals. Consult the original run's paired intervals and describe their conditional bootstrap limitations before claiming improvement.
- `event_analysis.csv`: every event, scores, rank ranges, expected 5% review inclusion, and retrospective risk increase.
- `error_summary.csv`: counts and expected review weights. Categories overlap. Partially reviewed boundary ties can appear among misses and reviews.
- `case_studies.csv`: up to ten examples per model/category, selected by largest retrospective log-risk increase, then event ID. These are illustrative selected cases, not a representative sample.
- `history_rank_changes.csv`, when available: the 30 largest absolute B-to-D average-rank changes. Positive gain means history ranked the event earlier; it does not establish improved performance.
- `ranking_demo.html`: open locally in a browser; no server or network is required. Later outcomes are hidden initially, but are embedded in the file and can be revealed for analysis. They never affect ordering or review weights.

## Interpret carefully

Late threshold crossing means latest visible log-risk < -6 and final log-risk >= -6; it is a retrospective definition, not an extra model feature. Neither this classification nor a selected example explains why a prediction changed. Low final endpoints do not establish unnecessary reviews. Model outputs estimate the final-message label, not physical collision probability. This cohort was already explored before the proposal; a rerun is not an independent test or evidence of progress by itself.

The exporter reads saved predictions without refitting models or changing the source run. Point estimates are recomputed from predictions, not copied from potentially stale summary CSVs. No model, cutoff, or decision threshold is selected by this export.
"""
    (output / "REPORT_GUIDE.md").write_text(text)
    provenance = {"source_run": run.name, "source_manifest_sha256": sha256(run / "manifest.json"),
                  "prediction_sha256": manifest["prediction_sha256"], "scheme": manifest["scheme"],
                  "primary_cutoff": primary, "source_completed_utc": manifest["completed_utc"],
                  "evidence_status": NOTICE,
                  "export_source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}}
    provenance["output_sha256"] = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    (output / "export_manifest.json").write_text(json.dumps(provenance, indent=2))
    return output
