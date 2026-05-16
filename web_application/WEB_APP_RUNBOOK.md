## Web App Runbook

## Local Run

### Prerequisite: Git LFS

Model files (`*.keras`) are tracked in Git LFS. Before running, ensure the real binaries are present:

```bash
git lfs version        # confirm git-lfs is installed
git lfs pull           # download model binaries (replaces LFS pointer stubs)
```

If `git lfs` is not installed, download it from https://git-lfs.github.com/ and run `git lfs install` once.

---

1. Create/activate your Python environment (recommended: `dogweb311`).
2. Verify your active interpreter before installing:

```bash
python --version
python -m pip --version
```

3. Install dependencies using `python -m pip` (not plain `pip`):

```bash
python -m pip install --upgrade pip
python -m pip install -r web_application/requirements.txt
```

If you are using conda and activation is inconsistent, run directly in the environment:

```bash
conda run -n dogweb311 python -m pip install -r web_application/requirements.txt
```

4. Start Streamlit:

```bash
python -m streamlit run web_application/streamlit_app.py
```

5. Open the local URL shown in terminal (usually http://localhost:8501).

## Troubleshooting: TensorFlow Install

If you see "No matching distribution found for tensorflow==2.18.0", your interpreter is likely Python 3.13.

1. Check interpreter and pip target:

```bash
python --version
python -m pip --version
```

2. If needed, switch to `dogweb311` in VS Code and rerun install using `python -m pip`.
3. Requirements are now Python-version aware:

- Python < 3.13 -> `tensorflow==2.18.0`
- Python >= 3.13 -> `tensorflow==2.20.0`

4. If pip shows a dependency resolver warning such as `fastai ... requires torchvision`, this is non-fatal for this web app unless pip ends with a hard failure and no `Successfully installed` message.

## Clean Environment (Recommended)

Use an isolated conda environment to avoid unrelated package conflicts from the base environment.

1. `conda create -n dogweb311 python=3.11 -y`
2. `conda run -n dogweb311 python -m pip install --upgrade pip`
3. `conda run -n dogweb311 python -m pip install -r web_application/requirements.txt`
4. `conda run -n dogweb311 python -m streamlit run web_application/streamlit_app.py`

## Streamlit Cloud Publish

1. Push current branch to GitHub.
2. In Streamlit Cloud, create a new app from repository `gsonego/Final_Dissertation`.
3. Set main file path to `web_application/streamlit_app.py`.
4. Deploy and test with jpg/jpeg/png uploads up to 5 MB.
5. Streamlit Cloud Python is pinned in `web_application/runtime.txt`.

## MVP Acceptance Checks

1. Upload valid jpg/png and verify top-1 + top-3 outputs.
2. Verify all three models (MobileNetV2, EfficientNetB0, ResNet50) render side by side for the same upload.
3. Verify the consensus summary and comparison table reflect the displayed predictions.
4. Verify graceful error for invalid type and oversized files.
5. Verify model metadata and UTC inference timestamp are shown.
6. Verify limitation disclaimer is visible with results.
