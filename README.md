# RetinaView — Explainable DR Screening

MERN application for diabetic-retinopathy screening. React provides the clinician dashboard; Express + MongoDB manage accounts and screening history; the Python service owns all ML inference and model artifacts.

## Architecture

```text
React client → Express API + MongoDB → Python inference service → final model artifacts
                         ↕
                 auth, patient metadata, history, report links
```

## Download from Drive

Place these **without changing their filenames** in `ml-service/artifacts/` (they are deliberately git-ignored):

1. `aptos_efficientnetb0_final.keras` — 5-grade classifier
2. `idrid_lesion_unet_final.weights.h5` — lesion-segmentation weights
3. `run_summary.json` — calibrated temperature, thresholds, and final metrics
4. The U-Net model-definition source used during training (for example `unet_model.py`). The architecture must exactly match the checkpoint before calling `load_weights`.

Optional: `Final_DR_Screening_Demo.ipynb` and `outputs_v3_fixed/` are useful for your records, but the deployed service does not need them. If the trained assets are not present, `DEMO_MODE=true` supplies deterministic demo responses only—never use that mode for real screening.

## Run locally

Prerequisites: Node 20+, Python 3.10+, MongoDB (local or Atlas).

```bash
copy server\\.env.example server\\.env
copy client\\.env.example client\\.env
copy ml-service\\.env.example ml-service\\.env
npm install
npm run install:all
python -m venv .venv
.venv\\Scripts\\activate
pip install -r ml-service/requirements.txt
uvicorn app:app --app-dir ml-service --reload --port 8000
npm run dev
```

Open `http://localhost:5173`. Create an account, then upload a retinal image. The API is at port 5000, and inference runs at port 8000.

Set `DEMO_MODE=false` only after copying all final artifacts and the exact U-Net architecture into `model_adapter.py`. The service intentionally refuses to guess a U-Net architecture from a weights-only file.

## Verify real inference before enabling it

With TensorFlow installed in the active Python environment, run this from the project root:

```bash
$env:DEMO_MODE='false'
python ml-service/smoke_test.py
```

Only if it prints `REAL SMOKE TEST PASSED`, set `DEMO_MODE=false` in `ml-service/.env` and start the ML service. In real mode, a missing or incompatible artifact fails startup explicitly; it never quietly returns fabricated predictions.

## Validated prototype results

| Measure | Result |
| --- | ---: |
| APTOS 5-class accuracy | 77.27% |
| Quadratic weighted kappa | 0.8202 |
| Referable-DR AUC | 0.9699 |
| Sensitivity / specificity | 90.58% / 88.99% |
| IDRiD lesion Dice / IoU | 0.5435 / 0.3732 |

This is a research/educational decision-support prototype, not a clinically validated diagnostic device. U-Net lesion masks are the primary explanation; Grad-CAM is auxiliary attention and should not be described as precise lesion localization. Messidor inference was demonstrated on 1,748 images, but the supplied CSV had no DR labels, so external accuracy could not be measured.
