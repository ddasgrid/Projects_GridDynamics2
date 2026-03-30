# Deploy `app2.py` (Streamlit)

This guide deploys your app so reviewers can run it directly in a browser.

## 1) Choose where checkpoint weights come from

You have two options:

### Option A: Local checkpoint in repo (simple, but large repo)
- Keep at least one supported `.pth` file in the repo root, e.g.:
  - `final_sota_visual_entailment3.pth`

### Option B: Download checkpoint from Hugging Face (recommended)
- Upload one checkpoint to a Hugging Face model repo.
- Set two environment variables in deployment:
  - `VE_HF_REPO_ID` (example: `your-username/ve-checkpoints`)
  - `VE_HF_FILENAME` (example: `final_sota_visual_entailment3.pth`)

`app2.py` now supports auto-download from HF cache when these are set.

---

## 2) Push code to GitHub

```bash
git add app2.py requirements.txt DEPLOY_APP2.md
git commit -m "Prepare app2 Streamlit deployment"
git push
```

---

## 3) Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io
2. Click **New app**
3. Select your GitHub repo + branch
4. Set **Main file path** to:
   - `app2.py`
5. In **Advanced settings**:
   - Python version: `3.11` (recommended)
   - (Optional) add environment variables for HF checkpoint:
     - `VE_HF_REPO_ID`
     - `VE_HF_FILENAME`
6. Click **Deploy**

---

## 4) Local smoke test before deploy

```bash
streamlit run app2.py
```

If using HF checkpoint locally:

```bash
export VE_HF_REPO_ID="your-username/ve-checkpoints"
export VE_HF_FILENAME="final_sota_visual_entailment3.pth"
streamlit run app2.py
```

---

## 5) Notes for reviewers

- The app requires model + tokenizer/image-processor download on first launch; first startup is slower.
- If no weights appear in the dropdown:
  - verify local `.pth` exists, or
  - verify `VE_HF_REPO_ID` and `VE_HF_FILENAME` are correct.
- The dropdown now shows checkpoint **file name** (cleaner UI), while loading from full path/HF cache internally.

