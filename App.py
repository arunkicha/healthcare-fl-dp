"""
Risk-Aware Privacy Preservation for Real-Time Health Analytics in EHR Systems
Working demo: Centralized baseline -> FedAvg -> Two-Tier DP-FedAvg

NOTE ON DATA: This demo generates synthetic EHR-style records that mirror the
structure of the project's real MIMIC-III Clinical Database Demo pipeline
(12 crosswalked vital-sign features + stigma-sensitive flags, 100-patient
scale, one deliberate outlier record). Real MIMIC-III data requires
PhysioNet/CITI credentialing and is not bundled here, so this demo lets you
show the actual mechanism (FedAvg, two-tier DP noise allocation, high-stigma-
wins overlap rule, outlier exclusion) running end-to-end and producing
results you can interpret the same way as the real experiments.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Tiered-DP Federated EHR Analytics", layout="wide")

RNG_SEED_DEFAULT = 42

NON_SENSITIVE_FEATURES = [
    "heart_rate", "resp_rate", "sbp", "dbp", "map", "spo2",
    "temp_c", "glucose", "wbc", "creatinine", "lactate", "urine_output"
]
STIGMA_SENSITIVE_FEATURES = [
    "substance_use_flag", "mental_health_flag", "hiv_status_flag", "housing_instability_flag"
]
ALL_FEATURES = NON_SENSITIVE_FEATURES + STIGMA_SENSITIVE_FEATURES


# ----------------------------- Data generation -----------------------------

def generate_synthetic_ehr(n_patients=100, seed=RNG_SEED_DEFAULT, include_outlier=True):
    rng = np.random.default_rng(seed)
    n = n_patients - 1 if include_outlier else n_patients

    vitals = pd.DataFrame({
        "heart_rate": rng.normal(85, 15, n),
        "resp_rate": rng.normal(18, 4, n),
        "sbp": rng.normal(120, 18, n),
        "dbp": rng.normal(75, 10, n),
        "map": rng.normal(90, 12, n),
        "spo2": rng.normal(96, 3, n),
        "temp_c": rng.normal(37.0, 0.8, n),
        "glucose": rng.normal(130, 40, n),
        "wbc": rng.normal(9, 3, n),
        "creatinine": rng.normal(1.1, 0.5, n),
        "lactate": rng.normal(1.8, 1.0, n),
        "urine_output": rng.normal(1500, 500, n),
    })

    flags = pd.DataFrame({
        "substance_use_flag": rng.binomial(1, 0.18, n),
        "mental_health_flag": rng.binomial(1, 0.22, n),
        "hiv_status_flag": rng.binomial(1, 0.06, n),
        "housing_instability_flag": rng.binomial(1, 0.15, n),
    })

    df = pd.concat([vitals, flags], axis=1)

    # Build a risk label correlated with a mix of vitals + sensitive flags + noise,
    # so both tiers carry genuine signal (mirrors a realistic risk-prediction task).
    risk_score = (
        0.03 * (df["heart_rate"] - 85)
        + 0.05 * (df["resp_rate"] - 18)
        - 0.04 * (df["spo2"] - 96)
        + 0.02 * (df["lactate"] - 1.8) * 5
        + 0.6 * df["substance_use_flag"]
        + 0.5 * df["mental_health_flag"]
        + 0.9 * df["hiv_status_flag"]
        + 0.4 * df["housing_instability_flag"]
        + rng.normal(0, 1.0, n)
    )
    prob = 1 / (1 + np.exp(-(risk_score - risk_score.mean()) / risk_score.std()))
    df["high_risk"] = rng.binomial(1, prob)

    if include_outlier:
        # Deliberate extreme outlier record, analogous to the excluded Subject ID
        # in the real pipeline: implausible vitals that would distort FL aggregation.
        outlier = {f: 0 for f in ALL_FEATURES}
        outlier.update({
            "heart_rate": 220, "resp_rate": 48, "sbp": 240, "dbp": 140,
            "map": 180, "spo2": 62, "temp_c": 41.5, "glucose": 480,
            "wbc": 32, "creatinine": 6.8, "lactate": 12.0, "urine_output": 50,
            "substance_use_flag": 1, "mental_health_flag": 1,
            "hiv_status_flag": 0, "housing_instability_flag": 1,
            "high_risk": 1,
        })
        df = pd.concat([df, pd.DataFrame([outlier])], ignore_index=True)

    return df


def standardize(train_df, test_df, feature_cols):
    mu = train_df[feature_cols].mean()
    sigma = train_df[feature_cols].std().replace(0, 1)
    train_std = (train_df[feature_cols] - mu) / sigma
    test_std = (test_df[feature_cols] - mu) / sigma
    return train_std.values, test_std.values


# ----------------------------- Model training -----------------------------

def fresh_model(n_features, seed):
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                         learning_rate="optimal", random_state=seed)
    # Initialize weights via a single partial_fit call on tiny dummy data so
    # coef_/intercept_ exist and can be manipulated before real training.
    dummy_X = np.zeros((2, n_features))
    dummy_y = np.array([0, 1])
    clf.partial_fit(dummy_X, dummy_y, classes=np.array([0, 1]))
    clf.coef_ = np.zeros_like(clf.coef_)
    clf.intercept_ = np.zeros_like(clf.intercept_)
    return clf


def train_centralized(X_train, y_train, X_test, y_test, seed):
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                         max_iter=300, random_state=seed)
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_test, clf.decision_function(X_test))
    return auc


def partition_clients(X, y, n_clients, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    client_idx = np.array_split(idx, n_clients)
    return [(X[ci], y[ci]) for ci in client_idx if len(ci) > 1]


def clip_update(delta, clip_norm):
    norm = np.linalg.norm(delta)
    if norm > clip_norm:
        delta = delta * (clip_norm / norm)
    return delta


def gaussian_sigma(epsilon, delta, sensitivity):
    # Standard Gaussian mechanism calibration.
    return (sensitivity / epsilon) * np.sqrt(2 * np.log(1.25 / delta))


def train_fedavg(X_train, y_train, X_test, y_test, n_clients, n_rounds, local_epochs,
                  seed, dp_mode=False, eps_sensitive=None, eps_nonsensitive=None,
                  delta=1e-5, clip_norm=1.0, sensitive_idx=None, nonsensitive_idx=None):
    n_features = X_train.shape[1]
    global_clf = fresh_model(n_features, seed)
    clients = partition_clients(X_train, y_train, n_clients, seed)
    rng = np.random.default_rng(seed)

    history = []

    for rnd in range(n_rounds):
        coef_updates, intercept_updates = [], []

        for (cx, cy) in clients:
            if len(np.unique(cy)) < 2:
                continue  # skip clients without both classes this round
            local_clf = fresh_model(n_features, seed)
            local_clf.coef_ = global_clf.coef_.copy()
            local_clf.intercept_ = global_clf.intercept_.copy()

            for _ in range(local_epochs):
                local_clf.partial_fit(cx, cy)

            delta_coef = (local_clf.coef_ - global_clf.coef_).flatten()
            delta_intercept = local_clf.intercept_ - global_clf.intercept_

            delta_coef = clip_update(delta_coef, clip_norm)

            if dp_mode:
                # Two-tier noise: stigma-sensitive feature weights get noise
                # calibrated to a tighter (smaller) epsilon -> larger sigma.
                # Non-sensitive weights use a looser epsilon -> smaller sigma.
                sigma_sens = gaussian_sigma(eps_sensitive, delta, clip_norm)
                sigma_nonsens = gaussian_sigma(eps_nonsensitive, delta, clip_norm)
                noise = np.zeros_like(delta_coef)
                noise[nonsensitive_idx] = rng.normal(0, sigma_nonsens, len(nonsensitive_idx))
                noise[sensitive_idx] = rng.normal(0, sigma_sens, len(sensitive_idx))
                delta_coef = delta_coef + noise

            coef_updates.append(delta_coef)
            intercept_updates.append(delta_intercept)

        if not coef_updates:
            continue

        avg_coef_update = np.mean(coef_updates, axis=0).reshape(global_clf.coef_.shape)
        avg_intercept_update = np.mean(intercept_updates, axis=0)

        global_clf.coef_ = global_clf.coef_ + avg_coef_update
        global_clf.intercept_ = global_clf.intercept_ + avg_intercept_update

        auc = roc_auc_score(y_test, global_clf.decision_function(X_test))
        history.append(auc)

    final_auc = history[-1] if history else 0.5
    return final_auc, history


# ----------------------------- Streamlit UI -----------------------------

st.title("Risk-Aware Privacy Preservation for Real-Time Health Analytics in EHR Systems")
st.caption("Two-tier differential privacy for federated EHR risk prediction — working demo")

with st.sidebar:
    st.header("Experiment controls")
    seed = st.number_input("Random seed", value=RNG_SEED_DEFAULT, step=1)
    n_clients = st.slider("Number of FL clients", 3, 15, 10)
    n_rounds = st.slider("FL rounds", 5, 50, 20)
    local_epochs = st.slider("Local epochs per round", 1, 10, 3)
    st.markdown("---")
    st.subheader("Two-tier privacy budget")
    eps_sensitive = st.slider("ε — stigma-sensitive tier (tighter = smaller)", 0.1, 5.0, 0.5, 0.1)
    eps_nonsensitive = st.slider("ε — non-sensitive tier (looser = larger)", 0.5, 10.0, 4.0, 0.1)
    clip_norm = st.slider("Update clipping norm (sensitivity bound)", 0.1, 5.0, 1.0, 0.1)
    st.markdown("---")
    exclude_outlier = st.checkbox("Exclude flagged outlier record", value=True)
    run_multiseed = st.checkbox("Run multi-seed stability check (5 seeds)", value=False)

if eps_sensitive >= eps_nonsensitive:
    st.sidebar.warning(
        "Sensitive-tier ε should normally be smaller (tighter privacy) than the "
        "non-sensitive tier's ε. Results will still run, but this inverts the "
        "intended two-tier design."
    )


@st.cache_data
def get_data(seed, include_outlier):
    return generate_synthetic_ehr(n_patients=100, seed=seed, include_outlier=include_outlier)


df = get_data(seed, include_outlier=not exclude_outlier and True)
# When "exclude" is checked, we still generate the outlier but drop it before use,
# so the toggle demonstrates the effect rather than never producing one.
df_full = generate_synthetic_ehr(n_patients=100, seed=seed, include_outlier=True)
if exclude_outlier:
    df_used = df_full.iloc[:-1].reset_index(drop=True)
else:
    df_used = df_full

sensitive_idx = [ALL_FEATURES.index(f) for f in STIGMA_SENSITIVE_FEATURES]
nonsensitive_idx = [ALL_FEATURES.index(f) for f in NON_SENSITIVE_FEATURES]

train_df, test_df = train_test_split(
    df_used, test_size=0.2, stratify=df_used["high_risk"], random_state=seed
)
X_train, X_test = standardize(train_df, test_df, ALL_FEATURES)
y_train, y_test = train_df["high_risk"].values, test_df["high_risk"].values

tab1, tab2, tab3, tab4 = st.tabs(
    ["Dataset", "Run demo", "Results & comparison", "Multi-seed stability"]
)

with tab1:
    st.subheader("Synthetic EHR cohort (mirrors MIMIC-III demo pipeline structure)")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(df_used.head(15), width='stretch')
    with c2:
        st.metric("Patients", len(df_used))
        st.metric("Non-sensitive vitals", len(NON_SENSITIVE_FEATURES))
        st.metric("Stigma-sensitive features", len(STIGMA_SENSITIVE_FEATURES))
        st.metric("High-risk prevalence", f"{df_used['high_risk'].mean():.1%}")
    st.markdown(
        "**Non-sensitive tier:** " + ", ".join(NON_SENSITIVE_FEATURES)
    )
    st.markdown(
        "**Stigma-sensitive tier (high-stigma-wins on any overlap):** "
        + ", ".join(STIGMA_SENSITIVE_FEATURES)
    )

with tab2:
    st.subheader("Run all three stages")
    if st.button("Run centralized -> FedAvg -> Two-tier DP-FedAvg", type="primary"):
        with st.spinner("Training centralized baseline..."):
            cent_auc = train_centralized(X_train, y_train, X_test, y_test, seed)

        with st.spinner("Running FedAvg (no DP)..."):
            fedavg_auc, fedavg_hist = train_fedavg(
                X_train, y_train, X_test, y_test, n_clients, n_rounds, local_epochs, seed,
                dp_mode=False
            )

        with st.spinner("Running two-tier DP-FedAvg..."):
            dp_auc, dp_hist = train_fedavg(
                X_train, y_train, X_test, y_test, n_clients, n_rounds, local_epochs, seed,
                dp_mode=True, eps_sensitive=eps_sensitive, eps_nonsensitive=eps_nonsensitive,
                clip_norm=clip_norm, sensitive_idx=sensitive_idx, nonsensitive_idx=nonsensitive_idx
            )

        st.session_state["results"] = {
            "cent_auc": cent_auc, "fedavg_auc": fedavg_auc, "fedavg_hist": fedavg_hist,
            "dp_auc": dp_auc, "dp_hist": dp_hist,
        }
        st.success("Done — see the Results tab.")

with tab3:
    if "results" not in st.session_state:
        st.info("Run the demo in the 'Run demo' tab first.")
    else:
        r = st.session_state["results"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Centralized ROC-AUC", f"{r['cent_auc']:.3f}")
        c2.metric("FedAvg ROC-AUC", f"{r['fedavg_auc']:.3f}",
                   delta=f"{r['fedavg_auc'] - r['cent_auc']:+.3f}")
        c3.metric("Two-tier DP-FedAvg ROC-AUC", f"{r['dp_auc']:.3f}",
                   delta=f"{r['dp_auc'] - r['fedavg_auc']:+.3f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=r["fedavg_hist"], mode="lines+markers", name="FedAvg (no DP)"))
        fig.add_trace(go.Scatter(y=r["dp_hist"], mode="lines+markers", name="Two-tier DP-FedAvg"))
        fig.add_hline(y=r["cent_auc"], line_dash="dash", annotation_text="Centralized baseline",
                      line_color="gray")
        fig.update_layout(title="ROC-AUC over FL rounds", xaxis_title="Round",
                           yaxis_title="ROC-AUC", height=420)
        st.plotly_chart(fig, width='stretch')

        st.subheader("What this shows")
        gap_fedavg = r["cent_auc"] - r["fedavg_auc"]
        gap_dp = r["cent_auc"] - r["dp_auc"]
        st.markdown(f"""
- **Centralization gap:** FedAvg trails the centralized baseline by
  **{gap_fedavg:.3f} AUC**, the expected cost of learning from partitioned,
  non-pooled client data instead of one pooled dataset.
- **Privacy cost of the two-tier mechanism:** DP-FedAvg trails plain FedAvg by
  **{gap_dp - gap_fedavg:+.3f} AUC** (or beats it, if negative) — this is the
  utility cost of enforcing a tighter ε on stigma-sensitive features while
  keeping the vitals tier at a looser budget.
- **Interpretation:** a small drop here is the expected trade-off — you're
  buying stronger protection specifically for the features most likely to
  cause harm if leaked (substance use, mental health, HIV status, housing
  instability), while spending less privacy budget (and losing less utility)
  on routine vitals that carry lower re-identification/stigma risk.
- **If DP-FedAvg AUC ≈ or > FedAvg AUC** at low round counts: this mirrors the
  "unexpected accuracy increase under noise" pattern flagged in the actual
  project — plausible on a small cohort, but it needs the multi-seed check
  in the next tab before treating it as a real effect rather than a
  small-sample artifact.
""")

with tab4:
    st.subheader("Multi-seed stability check")
    st.caption(
        "Re-runs two-tier DP-FedAvg across multiple seeds to check whether "
        "any AUC gain over plain FedAvg holds up, or was a small-sample artifact."
    )
    if run_multiseed and st.button("Run 5-seed stability sweep"):
        seeds = [seed + i for i in range(5)]
        rows = []
        prog = st.progress(0.0)
        for i, s in enumerate(seeds):
            df_s = generate_synthetic_ehr(n_patients=100, seed=s, include_outlier=True)
            df_s_used = df_s.iloc[:-1].reset_index(drop=True) if exclude_outlier else df_s
            tr, te = train_test_split(df_s_used, test_size=0.2,
                                       stratify=df_s_used["high_risk"], random_state=s)
            Xtr, Xte = standardize(tr, te, ALL_FEATURES)
            ytr, yte = tr["high_risk"].values, te["high_risk"].values

            fa_auc, _ = train_fedavg(Xtr, ytr, Xte, yte, n_clients, n_rounds, local_epochs, s,
                                      dp_mode=False)
            dp_auc, _ = train_fedavg(Xtr, ytr, Xte, yte, n_clients, n_rounds, local_epochs, s,
                                      dp_mode=True, eps_sensitive=eps_sensitive,
                                      eps_nonsensitive=eps_nonsensitive, clip_norm=clip_norm,
                                      sensitive_idx=sensitive_idx, nonsensitive_idx=nonsensitive_idx)
            rows.append({"seed": s, "FedAvg": fa_auc, "Two-tier DP-FedAvg": dp_auc})
            prog.progress((i + 1) / len(seeds))

        res_df = pd.DataFrame(rows)
        st.dataframe(res_df, width='stretch')

        fig2 = go.Figure()
        fig2.add_trace(go.Box(y=res_df["FedAvg"], name="FedAvg"))
        fig2.add_trace(go.Box(y=res_df["Two-tier DP-FedAvg"], name="Two-tier DP-FedAvg"))
        fig2.update_layout(title="AUC spread across 5 seeds", yaxis_title="ROC-AUC", height=400)
        st.plotly_chart(fig2, width='stretch')

        mean_gap = (res_df["Two-tier DP-FedAvg"] - res_df["FedAvg"]).mean()
        std_gap = (res_df["Two-tier DP-FedAvg"] - res_df["FedAvg"]).std()
        st.markdown(f"""
**Inference:** mean gap (DP-FedAvg − FedAvg) across seeds = **{mean_gap:+.3f}**,
std = **{std_gap:.3f}**. If this gap is small and its sign flips across seeds,
any single-seed "DP improves accuracy" result is noise, not a real effect —
report the distribution, not one lucky seed, in the paper's results section.
""")
    elif not run_multiseed:
        st.info("Enable 'Run multi-seed stability check' in the sidebar first.")