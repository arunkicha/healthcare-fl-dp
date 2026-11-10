"""Streamlit dashboard for the tiered-DP federated EHR project.

Every number shown is read from results/metrics/*.csv, which are produced by the
full-scale MIMIC-III pipeline (src/models/*, src/privacy/*). Only aggregate
results are displayed; no patient-level MIMIC data is read or shown.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODELS = ROOT / "models"
METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"
MODELING_TABLE = ROOT / "data" / "processed" / "modeling_table.csv"

REQUIRED = [
    "centralized_baseline.csv",
    "federated_no_dp.csv",
    "dp_federated_multiseed.csv",
    "dp_federated_epsilons.csv",
    "mia_multiseed.csv",
    "mia_validity_check.csv",
    "noise_ablation.csv",
]

GROUPS = ["overall", "stigma_sensitive", "non_sensitive"]
GROUP_LABELS = {
    "overall": "Overall",
    "stigma_sensitive": "Stigma-sensitive",
    "non_sensitive": "Non-sensitive",
}
GROUP_COLORS = {
    "overall": "#4c72b0",
    "stigma_sensitive": "#c44e52",
    "non_sensitive": "#55a868",
}
STAGES = ["noise0_sanity_check", "flat_dp", "tiered"]
STAGE_LABELS = {
    "noise0_sanity_check": "No-noise sanity check",
    "flat_dp": "Flat DP (noise 0.5 everywhere)",
    "tiered": "Tiered DP (noise 1.0 sensitive / 0.5 other)",
}
CONFIG_ORDER = ["no_dp", "noise0_sanity_check", "flat_dp", "tiered", "tiered_strong"]
CONFIG_LABELS = {
    "no_dp": "No DP (plain training)",
    "noise0_sanity_check": "Clipping only (noise 0)",
    "flat_dp": "Flat DP",
    "tiered": "Tiered DP",
    "tiered_strong": "Tiered DP (strong)",
}
ATTACK_LABELS = {
    "reference": "Reference-model attack (strongest)",
    "calibrated": "Class-calibrated loss attack",
    "loss": "Plain loss-threshold attack",
}
FALLBACK_STATS = {
    "patients": 46519,
    "admissions": 58961,
    "sensitive": 16622,
    "non_sensitive": 42339,
    "deaths": 5854,
}

st.set_page_config(
    page_title="Risk-Aware Federated DP on MIMIC-III",
    layout="wide",
)


@st.cache_data
def load_results():
    missing = [f for f in REQUIRED if not (METRICS / f).exists()]
    if missing:
        return None, missing
    data = {f[:-4]: pd.read_csv(METRICS / f) for f in REQUIRED}
    data["dp_federated_multiseed"]["stage"] = data["dp_federated_multiseed"][
        "stage"
    ].str.replace("dp_federated_", "", regex=False)
    return data, []


@st.cache_data
def load_dataset_stats():
    if not MODELING_TABLE.exists():
        return FALLBACK_STATS, "recorded from the dataset.py run"
    try:
        cols = pd.read_csv(MODELING_TABLE, nrows=0).columns
        needed = [c for c in ["subject_id", "final_tier", "hospital_expire_flag"] if c in cols]
        if len(needed) < 3:
            return FALLBACK_STATS, "recorded from the dataset.py run"
        df = pd.read_csv(MODELING_TABLE, usecols=needed)
        return {
            "patients": int(df["subject_id"].nunique()),
            "admissions": int(len(df)),
            "sensitive": int((df["final_tier"] == "stigma_sensitive").sum()),
            "non_sensitive": int((df["final_tier"] == "non_sensitive").sum()),
            "deaths": int(df["hospital_expire_flag"].sum()),
        }, "read from data/processed/modeling_table.csv (aggregates only)"
    except Exception:
        return FALLBACK_STATS, "recorded from the dataset.py run"


def mean_std(df, by, value):
    out = df.groupby(by)[value].agg(["mean", "std"]).reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out


def paired_tests(multiseed):
    pivot = multiseed.pivot_table(
        index=["group", "seed"], columns="stage", values="roc_auc"
    )
    pairs = [
        ("flat_dp", "tiered"),
        ("noise0_sanity_check", "flat_dp"),
        ("noise0_sanity_check", "tiered"),
    ]
    rows = []
    for group in GROUPS:
        sub = pivot.loc[group]
        for a, b in pairs:
            diff = sub[a] - sub[b]
            t_p = stats.ttest_rel(sub[a], sub[b]).pvalue
            try:
                w_p = stats.wilcoxon(sub[a], sub[b]).pvalue
            except ValueError:
                w_p = float("nan")
            rows.append(
                {
                    "group": GROUP_LABELS[group],
                    "comparison": f"{STAGE_LABELS[a].split(' (')[0]}  vs  {STAGE_LABELS[b].split(' (')[0]}",
                    "mean AUC difference": diff.mean(),
                    "paired t-test p": t_p,
                    "Wilcoxon p": w_p,
                }
            )
    return pd.DataFrame(rows)


data, missing = load_results()

st.title("Risk-Aware Privacy Preservation for Real-Time Health Analytics in EHR Systems")
st.caption(
    "Tiered differential privacy for federated EHR mortality prediction. "
    "All results below come from the full MIMIC-III v1.4 pipeline."
)

if data is None:
    st.error(
        "Result files not found in results/metrics/: "
        + ", ".join(missing)
        + ". Run the pipeline first (see README), then reload."
    )
    st.stop()

multiseed = data["dp_federated_multiseed"]
epsilons = data["dp_federated_epsilons"]
mia = data["mia_multiseed"]
validity = data["mia_validity_check"]
ablation = data["noise_ablation"]
centralized = data["centralized_baseline"]
federated = data["federated_no_dp"]

(
    tab_overview,
    tab_utility,
    tab_privacy,
    tab_tradeoff,
    tab_attack,
    tab_stress,
    tab_rt,
    tab_stats,
) = st.tabs(
    [
        "Overview",
        "Utility",
        "Privacy budget (epsilon)",
        "Privacy-utility tradeoff",
        "Membership inference attack",
        "Attack stress test",
        "Real-time scoring",
        "Statistics",
    ]
)

with tab_overview:
    st.subheader("The idea")
    st.markdown(
        "Federated learning keeps raw records at each hospital, but trained models can "
        "still leak whether a person was in the training data. Differential privacy (DP-SGD) "
        "bounds that leakage by clipping each patient's gradient and adding noise. "
        "Standard DP applies the same noise to everyone. **Tiered DP** splits each hospital's "
        "patients into a *stigma-sensitive* tier (for example psychiatric or substance-use "
        "diagnoses) and a *non-sensitive* tier, and trains each with its own noise level, "
        "so the data that is most harmful to leak gets the strongest guarantee."
    )

    ds, source = load_dataset_stats()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Patients", f"{ds['patients']:,}")
    c2.metric("Admissions", f"{ds['admissions']:,}")
    c3.metric("Stigma-sensitive admissions", f"{ds['sensitive']:,}")
    c4.metric("Non-sensitive admissions", f"{ds['non_sensitive']:,}")
    c5.metric("In-hospital mortality", f"{ds['deaths'] / ds['admissions']:.1%}")
    test_rows = centralized[(centralized["split"] == "test") & (centralized["group"] == "overall")]
    test_n = int(test_rows["n"].iloc[0])
    st.caption(f"Dataset figures: {source}. Test set: {test_n:,} admissions.")

    st.subheader("Pipeline")
    st.markdown(
        "1. **Data**: MIMIC-III admissions, diagnoses and 12 first-day vital-sign features "
        "(CHARTEVENTS read in chunks).\n"
        "2. **Tiering**: ICD-9 diagnoses map each admission to a sensitivity tier.\n"
        "3. **Federation**: 10 stratified simulated hospitals, 20 FedAvg rounds.\n"
        "4. **Training**: centralized and federated baselines, then DP-SGD (Opacus) "
        "with three configurations and 5 seeds each.\n"
        "5. **Evaluation**: ROC-AUC per tier, epsilon accounting, membership inference "
        "attack with a validity check, paired significance tests, noise ablation."
    )

    with st.expander("Honest limitations"):
        st.markdown(
            "- One dataset (MIMIC-III), one model type (MLP) and one attack family (loss-threshold).\n"
            "- The main attack finds no leakage in any configuration, so empirical evidence of "
            "*reduced* leakage from tiering is not yet shown; the strongest evidence is the "
            "formal epsilon versus utility result.\n"
            "- The sensitive-diagnosis list is a judgment-based taxonomy.\n"
            "- Hospitals are simulated by stratified splitting, so real-world heterogeneity is "
            "not captured.\n"
            "- The non-sensitive tier keeps a loose guarantee (epsilon about 27.5).\n"
            "- Reported epsilon values cover one round of local training. A record that stays in "
            "training is used in every round, so the cumulative epsilon over 20 rounds is roughly "
            "5 to 6 times larger (the Attack stress test tab reports both)."
        )

with tab_utility:
    st.subheader("Predictive utility (ROC-AUC on the held-out test set)")

    rows = []
    cent = centralized[centralized["split"] == "test"].set_index("group")["roc_auc"]
    fed = federated.set_index("group")["roc_auc"]
    rows.append({"Stage": "Centralized (no privacy)", **{GROUP_LABELS[g]: f"{cent[g]:.4f}" for g in GROUPS}})
    rows.append({"Stage": "Federated, no DP", **{GROUP_LABELS[g]: f"{fed[g]:.4f}" for g in GROUPS}})
    summary = mean_std(multiseed, ["stage", "group"], "roc_auc")
    for stage in STAGES:
        row = {"Stage": STAGE_LABELS[stage]}
        for g in GROUPS:
            r = summary[(summary["stage"] == stage) & (summary["group"] == g)].iloc[0]
            row[GROUP_LABELS[g]] = f"{r['mean']:.4f} ± {r['std']:.4f}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "DP rows are mean ± std over 5 seeds. Baseline rows are single runs. The three DP "
        "configurations share one training setup, so compare them with each other; the "
        "baselines were trained with separate settings."
    )

    fig = go.Figure()
    for g in GROUPS:
        sub = summary[summary["group"] == g].set_index("stage").loc[STAGES]
        fig.add_bar(
            name=GROUP_LABELS[g],
            x=[STAGE_LABELS[s].split(" (")[0] for s in STAGES],
            y=sub["mean"],
            error_y=dict(type="data", array=sub["std"]),
            marker_color=GROUP_COLORS[g],
        )
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="ROC-AUC", range=[0.8, 0.9]),
        title="DP configurations by tier (y-axis zoomed to 0.80-0.90)",
    )
    st.plotly_chart(fig, width="stretch")

    drop = (
        summary[(summary["stage"] == "noise0_sanity_check") & (summary["group"] == "overall")]["mean"].iloc[0]
        - summary[(summary["stage"] == "tiered") & (summary["group"] == "overall")]["mean"].iloc[0]
    )
    st.info(
        f"Switching from no noise to DP costs about {drop:.3f} AUC overall. "
        "Flat DP and tiered DP are practically identical in utility."
    )

with tab_privacy:
    st.subheader("Formal privacy budget (epsilon, delta = 1e-5; smaller is stronger)")
    eps = epsilons.dropna(subset=["epsilon"])
    eps_summary = eps.groupby(["stage", "tier"])["epsilon"].mean().reset_index()

    fig = go.Figure()
    for tier in ["stigma_sensitive", "non_sensitive"]:
        sub = eps_summary[eps_summary["tier"] == tier].set_index("stage").reindex(["flat_dp", "tiered"])
        fig.add_bar(
            name=GROUP_LABELS[tier],
            x=[STAGE_LABELS[s].split(" (")[0] for s in sub.index],
            y=sub["epsilon"],
            marker_color=GROUP_COLORS[tier],
            text=[f"{v:.2f}" for v in sub["epsilon"]],
            textposition="outside",
        )
    fig.update_layout(barmode="group", yaxis_title="Epsilon (privacy budget spent)")
    st.plotly_chart(fig, width="stretch")

    flat_eps = eps_summary[(eps_summary["stage"] == "flat_dp") & (eps_summary["tier"] == "stigma_sensitive")]["epsilon"].iloc[0]
    tier_eps = eps_summary[(eps_summary["stage"] == "tiered") & (eps_summary["tier"] == "stigma_sensitive")]["epsilon"].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Sensitive tier, flat DP", f"{flat_eps:.2f}")
    c2.metric("Sensitive tier, tiered DP", f"{tier_eps:.2f}")
    c3.metric("Guarantee tightened by", f"{flat_eps / tier_eps:.1f}x")
    st.caption(
        "The no-noise sanity check has no finite epsilon (no privacy). Epsilon is identical "
        "across seeds because it is computed by the privacy accountant, not measured."
    )

with tab_tradeoff:
    st.subheader("Noise ablation: privacy versus utility for the sensitive tier")
    abl = (
        ablation.groupby("stigma_noise")
        .agg(
            eps=("epsilon_stigma", "mean"),
            auc=("auc_stigma_sensitive", "mean"),
            auc_std=("auc_stigma_sensitive", "std"),
            overall=("auc_overall", "mean"),
        )
        .reset_index()
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=abl["stigma_noise"], y=abl["eps"], name="Epsilon (left)",
            mode="lines+markers", line=dict(color="#c44e52"),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=abl["stigma_noise"], y=abl["auc"], name="Sensitive-tier ROC-AUC (right)",
            mode="lines+markers", line=dict(color="#4c72b0"),
            error_y=dict(type="data", array=abl["auc_std"].fillna(0)),
        ),
        secondary_y=True,
    )
    fig.update_xaxes(title="Stigma-sensitive noise multiplier")
    fig.update_yaxes(title="Epsilon (smaller = stronger privacy)", secondary_y=False)
    fig.update_yaxes(title="ROC-AUC", range=[0.80, 0.90], secondary_y=True)
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Try a setting**")
    chosen = st.select_slider(
        "Sensitive-tier noise multiplier",
        options=list(abl["stigma_noise"]),
        value=1.0 if 1.0 in list(abl["stigma_noise"]) else list(abl["stigma_noise"])[0],
    )
    row = abl[abl["stigma_noise"] == chosen].iloc[0]
    base = abl[abl["stigma_noise"] == 0.5].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Epsilon", f"{row['eps']:.2f}", f"{row['eps'] - base['eps']:+.2f} vs flat DP", delta_color="inverse")
    c2.metric("Sensitive-tier AUC", f"{row['auc']:.4f}", f"{row['auc'] - base['auc']:+.4f} vs flat DP")
    c3.metric("Tighter than flat DP", f"{base['eps'] / row['eps']:.1f}x")
    st.caption(
        "Sensitive-tier AUC stays near 0.850 while epsilon falls from about 78 to about 2.5. "
        "Error bars (std across 3 seeds) reflect seed differences; within a seed the AUC "
        "moves by well under 0.001 across the sweep. Utility may break at much larger noise, "
        "which this range does not reach."
    )

with tab_attack:
    st.subheader("Membership inference attack (loss-threshold)")
    main = mia.groupby(["stage", "subgroup"])["attack_balanced_accuracy"].mean().reset_index()
    vrows = validity.copy()
    vrows["stage"] = "validity_check"
    vrows = vrows.rename(columns={"attack_balanced_accuracy": "attack_balanced_accuracy"})[
        ["stage", "subgroup", "attack_balanced_accuracy"]
    ]
    combined = pd.concat([main, vrows], ignore_index=True)
    labels = {**{s: STAGE_LABELS[s].split(" (")[0] for s in STAGES}, "validity_check": "Validity check (memorizing model)"}
    combined["label"] = combined["stage"].map(labels)

    fig = px.bar(
        combined,
        x="label",
        y="attack_balanced_accuracy",
        color="subgroup",
        barmode="group",
        color_discrete_map=GROUP_COLORS,
        category_orders={"label": list(labels.values())},
        labels={"attack_balanced_accuracy": "Attack balanced accuracy", "label": ""},
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="chance (0.5)")
    fig.update_yaxes(range=[0.4, 0.85])
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        "- **Main models**: balanced accuracy is about 0.50 everywhere, so the attacker cannot "
        "tell training members from non-members.\n"
        "- **Validity check**: a deliberately memorizing model reaches about 0.77, proving the "
        "attack can detect leakage when it exists. The null result above is therefore meaningful.\n"
        "- Balanced accuracy (not raw accuracy) is used because the raw metric equalled the "
        "class ratio and reflected imbalance rather than leakage. The attacker picks the best "
        "threshold using labels, so values slightly above 0.50 are the expected noise floor."
    )
    st.dataframe(
        combined.assign(**{"attack_balanced_accuracy": combined["attack_balanced_accuracy"].round(4)})[
            ["label", "subgroup", "attack_balanced_accuracy"]
        ],
        hide_index=True,
        width="stretch",
    )

with tab_stress:
    st.subheader("Attack stress test: a small-hospital regime where the non-private model leaks")
    st.markdown(
        "At full scale the model generalises, so no attack finds leakage even without noise. "
        "Here a small hospital network trains a wider network on a few thousand admissions "
        "without weight decay, so it memorises. The same federated pipeline is run with no "
        "privacy, clipping only, flat DP and tiered DP, and attacked three ways. "
        "A useful defence should push attack success from the no-DP level back toward 0.5."
    )
    leaky_files = ["leaky_regime_attacks.csv", "leaky_regime_utility.csv", "leaky_regime_epsilons.csv"]
    if not all((METRICS / f).exists() for f in leaky_files):
        st.info("No stress-test results yet. Run: python -m src.privacy.leaky_regime")
    else:
        l_attacks = pd.read_csv(METRICS / leaky_files[0])
        l_util = pd.read_csv(METRICS / leaky_files[1])
        l_eps = pd.read_csv(METRICS / leaky_files[2])
        order = [c for c in CONFIG_ORDER if c in set(l_attacks["config"])]

        c1, c2 = st.columns(2)
        attack = c1.selectbox(
            "Attack", [a for a in ATTACK_LABELS if a in set(l_attacks["attack"])],
            format_func=lambda a: ATTACK_LABELS[a],
        )
        metric = c2.radio(
            "Metric", ["balanced_accuracy", "tpr_at_5pct_fpr"],
            format_func=lambda m: {"balanced_accuracy": "Balanced accuracy (0.5 = chance)",
                                   "tpr_at_5pct_fpr": "True-positive rate at 5% false-positive rate (0.05 = chance)"}[m],
            horizontal=True,
        )
        sub = l_attacks[l_attacks["attack"] == attack]
        agg = sub.groupby(["config", "group"])[metric].agg(["mean", "std"]).reset_index()
        agg["std"] = agg["std"].fillna(0.0)
        fig = go.Figure()
        for g in GROUPS:
            part = agg[agg["group"] == g].set_index("config").reindex(order)
            fig.add_bar(
                name=GROUP_LABELS[g], x=[CONFIG_LABELS[c] for c in order], y=part["mean"],
                error_y=dict(type="data", array=part["std"]), marker_color=GROUP_COLORS[g],
            )
        chance = 0.5 if metric == "balanced_accuracy" else 0.05
        fig.add_hline(y=chance, line_dash="dash", line_color="gray", annotation_text="chance")
        fig.update_layout(barmode="group", yaxis_title=metric.replace("_", " "))
        st.plotly_chart(fig, width="stretch")

        n_seeds = l_attacks["seed"].nunique()
        st.caption(
            f"Mean ± std over {n_seeds} seeds. Members are training admissions; non-members are "
            "held-out patients. The reference-model attack uses attacker-trained models as a "
            "difficulty baseline, which makes it stronger than a plain loss threshold."
        )

        st.markdown("**Does the non-private model actually overfit here?** (generalisation gap)")
        gap = l_util.groupby(["config", "group"])[["train_auc", "test_auc"]].mean().reset_index()
        gap["gap"] = gap["train_auc"] - gap["test_auc"]
        gap = gap[gap["group"] == "overall"].set_index("config").reindex(order).reset_index()
        gap["config"] = gap["config"].map(CONFIG_LABELS)
        st.dataframe(
            gap[["config", "train_auc", "test_auc", "gap"]].round(4),
            hide_index=True, width="stretch",
        )

        st.markdown("**Privacy budget in this regime** (per round vs cumulative over all rounds)")
        eps_view = l_eps.groupby(["config", "tier"])[
            ["epsilon_per_round_mean", "epsilon_cumulative_mean", "epsilon_cumulative_max"]
        ].mean().reset_index()
        eps_view = eps_view.dropna(subset=["epsilon_per_round_mean"])
        eps_view["config"] = eps_view["config"].map(CONFIG_LABELS)
        st.dataframe(eps_view.round(2), hide_index=True, width="stretch")
        st.caption(
            "Per-round epsilon is what the main pipeline reports; cumulative epsilon composes "
            "every round and is the honest total privacy loss for a record that stays in training."
        )

with tab_rt:
    st.subheader("Real-time risk scoring with the privately trained model")
    st.markdown(
        "Scores one admission at a time with the exported federated model. Training used tiered "
        "DP-SGD, so serving the model adds no extra privacy cost. Inputs below are typed in by "
        "you; no patient records are loaded."
    )
    try:
        from src.realtime.predictor import (
            FEATURE_RANGES, RiskPredictor, available_models, risk_band,
        )
        rt_ready = True
    except Exception as exc:
        rt_ready = False
        st.warning(f"Real-time module could not be loaded ({exc}). Install torch and rerun.")

    models_found = available_models(str(MODELS)) if rt_ready else []
    if rt_ready and not models_found:
        st.info(
            "No exported model found. Run: python -m src.realtime.export_model --config tiered "
            "(optionally also --config noise0_sanity_check for a non-private comparison)."
        )
    elif rt_ready:

        @st.cache_resource
        def get_predictor(name):
            return RiskPredictor(model_dir=str(MODELS), name=name)

        default_idx = models_found.index("tiered") if "tiered" in models_found else 0
        chosen = st.selectbox("Model", models_found, index=default_idx)
        predictor = get_predictor(chosen)
        base = predictor.default_record()

        with st.expander("Model card"):
            st.json(predictor.card)

        cols = st.columns(4)
        record = {}
        age_lo, age_hi = FEATURE_RANGES["age"]
        record["age"] = cols[0].number_input("age", age_lo, age_hi, float(base["age"]), 1.0)
        record["gender"] = cols[1].selectbox("gender", ["M", "F"])
        vitals = [c for c in predictor.feature_cols if c not in ("age", "gender")]
        for i, col in enumerate(vitals):
            lo, hi = FEATURE_RANGES[col]
            record[col] = cols[(i + 2) % 4].number_input(col, lo, hi, float(base[col]), 1.0)
        missing = st.multiselect("Treat as missing (imputed with training median)", vitals)
        for col in missing:
            record[col] = None

        result = predictor.predict(record)
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted in-hospital mortality risk", f"{result['risk']:.1%}")
        m2.metric("Risk band", result["band"])
        m3.metric("Scoring latency", f"{result['latency_ms']:.2f} ms")
        if result["imputed"]:
            st.caption("Imputed: " + ", ".join(result["imputed"]))
        for w in result["warnings"]:
            st.warning(w)

        if len(models_found) > 1:
            st.markdown("**Same input, every exported model**")
            rows = []
            for name in models_found:
                r = get_predictor(name).predict(record)
                rows.append({"model": name, "risk": f"{r['risk']:.1%}", "band": r["band"],
                             "latency_ms": round(r["latency_ms"], 2)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        st.markdown("**Live latency burst** (synthetic inputs drawn around the training medians)")
        n_burst = st.slider("Number of single-record scorings", 50, 2000, 300, 50)
        if st.button("Run burst"):
            rng = np.random.default_rng(0)
            lat = []
            for _ in range(n_burst):
                rec = {c: float(base[c]) * float(rng.normal(1.0, 0.15)) for c in vitals}
                rec["age"] = float(rng.uniform(25, 90))
                rec["gender"] = "M" if rng.random() < 0.5 else "F"
                lat.append(predictor.predict(rec)["latency_ms"])
            lat = np.array(lat)
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("p50", f"{np.percentile(lat, 50):.2f} ms")
            b2.metric("p95", f"{np.percentile(lat, 95):.2f} ms")
            b3.metric("p99", f"{np.percentile(lat, 99):.2f} ms")
            b4.metric("Throughput", f"{n_burst / (lat.sum() / 1000.0):,.0f} / s")
            st.plotly_chart(px.histogram(x=lat, nbins=40, labels={"x": "latency (ms)"}), width="stretch")

        stream_summary = METRICS / "realtime_latency.csv"
        if stream_summary.exists():
            st.markdown("**Stream replay of held-out admissions** (from src/realtime/stream_sim.py)")
            s = pd.read_csv(stream_summary).iloc[0]
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Admissions replayed", f"{int(s['n_scored']):,}")
            s2.metric("p95 latency", f"{s['latency_p95_ms']:.2f} ms")
            s3.metric("Within budget", f"{s['within_budget_pct']:.1f}% (<= {s['budget_ms']:.0f} ms)")
            s4.metric("AUC on stream", f"{s['auc_on_stream']:.3f}")

with tab_stats:
    st.subheader("Seed stability and significance")
    fig = px.box(
        multiseed.assign(Stage=multiseed["stage"].map(lambda s: STAGE_LABELS[s].split(" (")[0])),
        x="group",
        y="roc_auc",
        color="Stage",
        points="all",
        category_orders={"group": GROUPS},
        labels={"roc_auc": "ROC-AUC", "group": ""},
    )
    st.plotly_chart(fig, width="stretch")

    tests = paired_tests(multiseed)
    st.dataframe(
        tests.style.format(
            {
                "mean AUC difference": "{:.5f}",
                "paired t-test p": "{:.4f}",
                "Wilcoxon p": "{:.4f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Positive difference means the first configuration scored higher. Flat vs tiered "
        "differences are tiny and not significant for the sensitive tier. With 5 seeds the "
        "two-sided Wilcoxon p-value cannot go below 0.0625, so rely on the t-test for the "
        "no-noise comparisons."
    )

if FIGURES.exists() and any(FIGURES.glob("*.png")):
    with st.expander("Static figures generated by src/visualize.py"):
        for img in sorted(FIGURES.glob("*.png")):
            st.image(str(img), caption=img.name)
