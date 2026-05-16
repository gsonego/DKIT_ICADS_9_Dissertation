## Web Application Planning

## Goal

Plan a web application for later integration, without claiming implementation in the current dissertation version.

## Scope Boundary

- In scope now: planning only.
- Out of scope now: any dissertation claim that the web app is already built/tested.
- Revisit report text only after MVP is functional and validated.

## Proposed MVP

1. Input:

- Upload a dog image (jpg/jpeg/png), max 5 MB.

2. Inference:

- Load selected final model artifact (`model.keras`).
- Apply correct preprocessing for chosen backbone.
- Allow user-selectable model (`MobileNetV2` or `EfficientNetB0`).
- Produce top-1 prediction + confidence and top-3 probabilities.

3. Output:

- Predicted breed label.
- Confidence score.
- Top-3 probabilities.
- Model name and run used.
- Note: closed-set classifier limitations.

## Architecture Plan

1. Application stack:

- Streamlit app for MVP (single application layer for UI + inference).
- TensorFlow/Keras model loading at startup (cached resource).
- Local inference workflow triggered by upload action.

2. UI:

- Upload page with result panel.
- Basic validation and clear error messages.

3. Model handling:

- Config file mapping model run name -> preprocessing function -> class labels.
- Explicit support for `MobileNetV2` and `EfficientNetB0` artifacts.

## Non-Functional Requirements

1. Reproducibility:

- Same label mapping as notebook/report.
- Versioned model artifact path.

2. Reliability:

- Validate file types and size.
- Graceful failure if model/artifact is missing.

3. Transparency:

- Display model/run metadata and inference timestamp.

4. Privacy:

- No server-side image retention; process uploads in-memory only.

## Validation Plan

1. Functional tests:

- Known in-scope samples return expected labels at reasonable confidence.
- Out-of-scope samples still produce deterministic output with disclaimer.

2. Consistency checks:

- Compare app predictions with notebook inference for same images.

3. UX checks:

- Clear upload instructions and understandable output text.

## Integration Back to Dissertation (Later)

Only after MVP is implemented and validated, add a short subsection in report:

1. Purpose and scope of demo app.
2. Model used and known limitations.
3. 1-2 screenshots and one concise usage example.

## Open Decisions

Resolved decisions:

1. Framework and app style: Streamlit for MVP.
2. Deployment target: Streamlit Cloud with public URL.
3. Model selection policy: selectable model (`MobileNetV2` and `EfficientNetB0`).
4. Prediction display policy: top-3 probabilities + closed-set disclaimer.
5. Upload constraints: jpg/jpeg/png, max 5 MB.
6. Data handling: no server-side image storage.

## Deployment Note

- Publish directly from GitHub to Streamlit Cloud for fastest online availability.
- Keep a fallback path to FastAPI + Cloud Run if stronger API control or scaling is needed later.
