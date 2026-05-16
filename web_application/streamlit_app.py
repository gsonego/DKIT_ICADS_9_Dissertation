from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import streamlit as st

from web_app.inference import decode_and_prepare_image, load_model, predict_topk, validate_upload
from web_app.model_manifest import CLASS_IDS, CLASS_ID_TO_NAME, MAX_UPLOAD_MB, MODEL_MANIFEST

MODEL_ACCENTS = {
    "MobileNetV2": {"edge": "#0f766e", "surface": "#ecfeff", "chip": "#ccfbf1"},
    "EfficientNetB0": {"edge": "#1d4ed8", "surface": "#eff6ff", "chip": "#dbeafe"},
    "ResNet50": {"edge": "#9a3412", "surface": "#fff7ed", "chip": "#ffedd5"},
}

st.set_page_config(page_title="Dog Breed Classifier Demo", page_icon=":dog:", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        color: #0f172a;
    }
    [data-testid="stHeader"] {
        background: rgba(248, 250, 252, 0.82);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        padding-bottom: 3rem;
    }
    .hero-panel {
        background: linear-gradient(135deg, #182235 0%, #0f172a 58%, #163a2f 100%);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 24px;
        padding: 1.25rem 1.4rem;
        margin: 0.25rem 0 1.25rem 0;
        box-shadow: 0 16px 40px rgba(15, 23, 42, 0.18);
    }
    .hero-eyebrow {
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #93c5fd;
        margin-bottom: 0.4rem;
    }
    .hero-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.55rem;
    }
    .hero-copy {
        color: #dbeafe;
        line-height: 1.55;
        margin: 0;
    }
    .consensus-panel {
        background: linear-gradient(135deg, #f8fafc 0%, #dbeafe 40%, #dcfce7 100%);
        border: 1px solid rgba(30, 41, 59, 0.12);
        border-radius: 26px;
        padding: 1.5rem 1.6rem;
        margin: 1.25rem auto 1.4rem auto;
        text-align: center;
        color: #0f172a;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.14);
    }
    .consensus-panel.warning {
        background: linear-gradient(135deg, #fff7cc 0%, #fde68a 48%, #fbbf24 100%);
        border: 1px solid rgba(146, 64, 14, 0.18);
        box-shadow: 0 18px 45px rgba(217, 119, 6, 0.18);
    }
    .consensus-kicker {
        font-size: 0.82rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #0369a1;
        margin-bottom: 0.45rem;
    }
    .consensus-panel.warning .consensus-kicker {
        color: #92400e;
    }
    .consensus-title {
        font-size: 1.7rem;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 0.55rem;
    }
    .consensus-detail {
        font-size: 1rem;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    .consensus-meta {
        font-size: 0.92rem;
        color: #334155;
    }
    .side-panel {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 22px;
        padding: 0.9rem 1rem;
        margin-top: 0.1rem;
        margin-bottom: 0.9rem;
        color: #0f172a;
    }
    .side-panel strong {
        display: block;
        margin-bottom: 0.3rem;
    }
    .side-panel p, .side-panel li {
        color: #334155;
    }
    .side-panel p {
        margin: 0 0 0.55rem 0;
        line-height: 1.45;
    }
    .side-panel ul {
        margin: 0.35rem 0 0 1rem;
        padding: 0;
    }
    .side-panel li {
        margin-bottom: 0.28rem;
    }
    .compare-button-spacer {
        height: 0.55rem;
    }
    div[data-testid="stButton"] button {
        margin-top: 0.2rem;
    }
    div[data-testid="stAlertContainer"] {
        color: #0f172a;
    }
    div[data-testid="stAlertContainer"] p,
    div[data-testid="stAlertContainer"] span,
    div[data-testid="stAlertContainer"] label,
    div[data-testid="stAlertContainer"] div {
        color: #0f172a !important;
    }
    div[data-testid="stAlertContainer"] svg {
        fill: #0f172a;
        color: #0f172a;
    }
    div[data-testid="stAlertContainer"] [data-baseweb="notification"] {
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.24);
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
    }
    .overview-section {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 26px;
        padding: 1.3rem 1.35rem 1.15rem 1.35rem;
        margin-top: 1.2rem;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
    }
    .overview-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 20px;
        padding: 1rem 1.05rem;
        min-height: 150px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }
    .overview-label {
        font-size: 0.82rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 0.5rem;
    }
    .overview-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.35rem;
    }
    .overview-copy {
        color: #334155;
        line-height: 1.45;
    }
    .model-card {
        border-radius: 24px;
        padding: 1.15rem 1.15rem 1rem 1.15rem;
        border: 1px solid rgba(148, 163, 184, 0.22);
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
        min-height: 420px;
    }
    .model-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 0.95rem;
    }
    .model-card-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
    }
    .model-chip {
        border-radius: 999px;
        padding: 0.28rem 0.65rem;
        font-size: 0.78rem;
        font-weight: 700;
        color: #0f172a;
        white-space: nowrap;
    }
    .model-card-meta {
        color: #475569;
        font-size: 0.88rem;
        line-height: 1.5;
        margin-bottom: 1rem;
    }
    .model-top1-label {
        font-size: 0.82rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #475569;
        margin-bottom: 0.25rem;
    }
    .model-top1-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
        margin-bottom: 0.15rem;
    }
    .model-top1-confidence {
        font-size: 1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1rem;
    }
    .model-top3-heading {
        font-size: 0.9rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.7rem;
    }
    .model-top3-row {
        margin-bottom: 0.9rem;
    }
    .model-top3-text {
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        font-size: 0.92rem;
        color: #1e293b;
        margin-bottom: 0.35rem;
    }
    .model-top3-breed {
        font-weight: 600;
    }
    .model-top3-id {
        color: #64748b;
        font-weight: 500;
    }
    .model-bar-track {
        height: 10px;
        width: 100%;
        background: #e2e8f0;
        border-radius: 999px;
        overflow: hidden;
    }
    .model-bar-fill {
        height: 100%;
        border-radius: 999px;
    }
    .combined-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.7rem;
        background: #ffffff;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 18px;
        overflow: hidden;
    }
    .combined-table thead th {
        background: #e2e8f0;
        color: #0f172a;
        text-align: left;
        font-size: 0.84rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 0.85rem 0.95rem;
    }
    .combined-table tbody td {
        padding: 0.8rem 0.95rem;
        border-top: 1px solid #e2e8f0;
        color: #1e293b;
        background: #ffffff;
    }
    .combined-table tbody tr:nth-child(even) td {
        background: #f8fafc;
    }
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.7rem;
        background: #ffffff;
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }
    .comparison-table thead th {
        background: #dbeafe;
        color: #0f172a;
        text-align: left;
        font-size: 0.84rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 0.85rem 0.95rem;
    }
    .comparison-table tbody td {
        padding: 0.8rem 0.95rem;
        border-top: 1px solid #e2e8f0;
        color: #1e293b;
        background: #ffffff;
    }
    .comparison-table tbody tr:nth-child(even) td {
        background: #f8fafc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def run_comparison(data: bytes) -> tuple[list[dict[str, object]], str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    results = []

    for model_label, model_spec in MODEL_MANIFEST.items():
        started_at = perf_counter()
        model = get_cached_model(str(model_spec["artifact"]), model_spec["backbone"])
        image_tensor = decode_and_prepare_image(data=data, backbone=model_spec["backbone"])
        top1, top3 = predict_topk(
            model=model,
            image_tensor=image_tensor,
            class_ids=CLASS_IDS,
            class_id_to_name=CLASS_ID_TO_NAME,
            k=3,
        )
        results.append(
            {
                "label": model_label,
                "run_name": model_spec["run_name"],
                "backbone": model_spec["backbone"],
                "top1": top1,
                "top3": top3,
                "latency_ms": round((perf_counter() - started_at) * 1000, 1),
            }
        )

    return results, timestamp


def build_summary(results: list[dict[str, object]]) -> tuple[str, str, str]:
    winners: dict[str, int] = {}
    for result in results:
        breed = result["top1"]["breed"]
        winners[breed] = winners.get(breed, 0) + 1

    consensus_breed, votes = max(winners.items(), key=lambda item: item[1])
    if votes == len(results):
        headline = f"Consensus: all {len(results)} models predict {consensus_breed}."
        banner_state = "success"
    elif votes > 1:
        headline = f"Consensus: {consensus_breed} leads with {votes}/{len(results)} top-1 votes."
        banner_state = "warning"
    else:
        headline = "No consensus: each model picked a different top-1 breed."
        banner_state = "warning"

    most_confident = max(results, key=lambda result: result["top1"]["confidence"])
    detail = (
        f"Most confident model: {most_confident['label']} "
        f"({most_confident['top1']['confidence']:.2%} on {most_confident['top1']['breed']})."
    )
    return headline, detail, banner_state


def build_overview(results: list[dict[str, object]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    top1_breeds = [result["top1"]["breed"] for result in results]
    unique_top1 = sorted(set(top1_breeds))
    shared_top3 = sorted(
        set.intersection(*(set(row["breed"] for row in result["top3"]) for result in results))
    )
    average_confidence = sum(result["top1"]["confidence"] for result in results) / len(results)
    confidence_range = max(result["top1"]["confidence"] for result in results) - min(
        result["top1"]["confidence"] for result in results
    )

    summary_cards = [
        {
            "label": "Top-1 agreement",
            "value": f"{len(results) - len(unique_top1) + 1}/{len(results)} models",
            "copy": ", ".join(unique_top1),
        },
        {
            "label": "Average confidence",
            "value": f"{average_confidence:.2%}",
            "copy": f"Spread across models: {confidence_range:.2%}",
        },
        {
            "label": "Shared top-3 breeds",
            "value": ", ".join(shared_top3) if shared_top3 else "None",
            "copy": "Breeds appearing in every model's top-3 list.",
        },
    ]

    combined_rows = []
    breed_mentions: dict[str, dict[str, object]] = {}
    for result in results:
        for rank, row in enumerate(result["top3"], start=1):
            breed_summary = breed_mentions.setdefault(
                row["breed"],
                {"votes": 0, "best_rank": rank, "best_confidence": row["confidence"]},
            )
            breed_summary["votes"] += 1
            breed_summary["best_rank"] = min(breed_summary["best_rank"], rank)
            breed_summary["best_confidence"] = max(breed_summary["best_confidence"], row["confidence"])

    for breed, metrics in sorted(
        breed_mentions.items(),
        key=lambda item: (-item[1]["votes"], item[1]["best_rank"], -item[1]["best_confidence"]),
    ):
        combined_rows.append(
            {
                "Breed": breed,
                "Top-3 mentions": f"{metrics['votes']}/{len(results)} models",
                "Best rank": metrics["best_rank"],
                "Best confidence": f"{metrics['best_confidence']:.2%}",
            }
        )

    return summary_cards, combined_rows


def render_consensus_banner(headline: str, detail: str, timestamp: str, banner_state: str) -> None:
    banner_class = "consensus-panel" if banner_state == "success" else "consensus-panel warning"
    st.markdown(
        f"""
        <div class="{banner_class}">
            <div class="consensus-kicker">Model consensus</div>
            <div class="consensus-title">{headline}</div>
            <div class="consensus-detail">{detail}</div>
            <div class="consensus-meta">Inference timestamp: {timestamp}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(summary_cards: list[dict[str, str]], combined_rows: list[dict[str, str]]) -> None:
    st.markdown('<div class="overview-section">', unsafe_allow_html=True)
    st.markdown("**Overall comparison view**")
    overview_columns = st.columns(len(summary_cards), gap="large")
    for column, card in zip(overview_columns, summary_cards):
        with column:
            st.markdown(
                f"""
                <div class="overview-card">
                    <div class="overview-label">{card['label']}</div>
                    <div class="overview-value">{card['value']}</div>
                    <div class="overview-copy">{card['copy']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    table_rows = "".join(
        f"<tr><td>{row['Breed']}</td><td>{row['Top-3 mentions']}</td><td>{row['Best rank']}</td><td>{row['Best confidence']}</td></tr>"
        for row in combined_rows
    )
    st.markdown("**Combined breed view**")
    st.markdown(
        f"""
        <table class="combined-table">
            <thead>
                <tr>
                    <th>Breed</th>
                    <th>Top-3 mentions</th>
                    <th>Best rank</th>
                    <th>Best confidence</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_card(result: dict[str, object]) -> None:
    top1 = result["top1"]
    top3 = result["top3"]
    accent = MODEL_ACCENTS[result["label"]]
    top3_rows = "".join(
        f"""
        <div class="model-top3-row">
            <div class="model-top3-text">
                <span class="model-top3-breed">{row['breed']} <span class="model-top3-id">({row['class_id']})</span></span>
                <span>{row['confidence']:.2%}</span>
            </div>
            <div class="model-bar-track">
                <div class="model-bar-fill" style="width: {row['confidence'] * 100:.2f}%; background: {accent['edge']};"></div>
            </div>
        </div>
        """
        for row in top3
    )

    st.markdown(
        f"""
        <div class="model-card" style="background: linear-gradient(180deg, {accent['surface']} 0%, #ffffff 100%); border-top: 5px solid {accent['edge']};">
            <div class="model-card-header">
                <div class="model-card-title">{result['label']}</div>
                <div class="model-chip" style="background: {accent['chip']};">{result['backbone']}</div>
            </div>
            <div class="model-card-meta">Run: {result['run_name']}<br>Inference: {result['latency_ms']} ms</div>
            <div class="model-top1-label">Top-1 prediction</div>
            <div class="model-top1-value">{top1['breed']}</div>
            <div class="model-top1-confidence">Confidence: {top1['confidence']:.2%}</div>
            <div class="model-top3-heading">Top-3 probabilities</div>
            {top3_rows}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.title("Dog Breed Classifier Demo")
st.caption("Dissertation MVP demo: compare 3 closed-set dog breed classifiers over the same upload.")

st.markdown(
    """
    <div class="hero-panel">
        <div class="hero-eyebrow">Comparison dashboard</div>
        <div class="hero-title">See how three backbones respond to the same image.</div>
        <p class="hero-copy">Upload one image, run all models together, and inspect where the predictions align, diverge, or cluster around the same breeds.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "This app is a closed-set classifier. It always predicts one of the 10 trained breeds, "
    "including for out-of-scope images. Use outputs as model estimates, not as definitive labels."
)

uploaded = st.file_uploader(
    "Upload a dog image",
    type=["jpg", "jpeg", "png"],
    help=(
        f"Accepted formats: jpg/jpeg/png. Maximum size: {MAX_UPLOAD_MB} MB. "
        "The app will run MobileNetV2, EfficientNetB0, and ResNet50 side by side."
    ),
)

st.caption("Privacy: uploads are processed in-memory for inference and are not stored server-side by this app.")


@st.cache_resource(show_spinner=False)
def get_cached_model(model_path_str: str, backbone: str):
    return load_model(model_path=Path(model_path_str), backbone=backbone)


if uploaded is not None:
    data = uploaded.getvalue()
    validation_error = validate_upload(uploaded.name, data)
    if validation_error:
        st.error(validation_error)
    else:
        preview_col, notes_col = st.columns([1.1, 1], gap="large")
        with preview_col:
            st.image(data, caption="Uploaded image", use_container_width=True)
        with notes_col:
            st.markdown(
                """
                <div class="side-panel">
                    <strong>Comparison mode</strong>
                    <p>All three models score the same image independently so you can inspect agreement, confidence, and rank-order differences.</p>
                    <strong>What to look for</strong>
                    <ul>
                        <li>Agreement on the top-1 breed</li>
                        <li>Confidence gaps between models</li>
                        <li>Whether the same breed appears across the top-3 lists</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="compare-button-spacer"></div>', unsafe_allow_html=True)
            run_comparison_now = st.button("Compare all models", type="primary", use_container_width=True)

        if run_comparison_now:
            try:
                with st.spinner("Loading models and running comparison..."):
                    results, timestamp = run_comparison(data)
            except Exception as exc:
                with notes_col:
                    st.error(f"Inference failed: {exc}")
            else:
                headline, detail, banner_state = build_summary(results)
                summary_cards, combined_rows = build_overview(results)

                render_consensus_banner(headline, detail, timestamp, banner_state)

                result_columns = st.columns(len(results), gap="large")
                for column, result in zip(result_columns, results):
                    with column:
                        render_model_card(result)

                render_overview(summary_cards, combined_rows)

                st.markdown("**Per-model comparison**")
                comparison_rows = [
                    {
                        "Model": result["label"],
                        "Top-1": result["top1"]["breed"],
                        "Confidence": f"{result['top1']['confidence']:.2%}",
                        "Second choice": result["top3"][1]["breed"],
                        "Third choice": result["top3"][2]["breed"],
                        "Inference (ms)": result["latency_ms"],
                    }
                    for result in results
                ]
                comparison_table_rows = "".join(
                    f"<tr><td>{row['Model']}</td><td>{row['Top-1']}</td><td>{row['Confidence']}</td><td>{row['Second choice']}</td><td>{row['Third choice']}</td><td>{row['Inference (ms)']}</td></tr>"
                    for row in comparison_rows
                )
                st.markdown(
                    f"""
                    <table class="comparison-table">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Top-1</th>
                                <th>Confidence</th>
                                <th>Second choice</th>
                                <th>Third choice</th>
                                <th>Inference (ms)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {comparison_table_rows}
                        </tbody>
                    </table>
                    """,
                    unsafe_allow_html=True,
                )

                st.warning(
                    "Limitation: confidence values are not calibrated probabilities and can be misleading "
                    "for out-of-scope images."
                )
else:
    st.write("Upload an image to begin.")
