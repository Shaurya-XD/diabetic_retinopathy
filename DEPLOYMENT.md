# EyeZen production deployment: Vercel + Railway

## Architecture

Vercel hosts the React/Vite application and its Node/Express API. MongoDB Atlas stores users and account-owned screening records. Private retinal inputs, U-Net overlays, Grad-CAM images, and PDFs are stored in Vercel Blob. Railway hosts only the persistent FastAPI/TensorFlow V4 ML service.

The browser never calls Railway. It uploads the retinal image directly to a scoped Vercel Blob PUT URL, then calls Express. Express creates scoped read/write Blob URLs, sends them with patient metadata and `X-EyeZen-Internal-Secret` to Railway, and stores the canonical result in Atlas. Railway has no MongoDB, JWT, or Blob token access.

> Railway needs meaningful memory for TensorFlow, EfficientNet, and the U-Net. Start with at least 4 GB of service memory; do not use a 512 MB service.

## Local development

1. In `server/.env`, configure `MONGODB_URI`, `JWT_SECRET`, `ML_SERVICE_URL=http://127.0.0.1:8000`, and an `INTERNAL_ML_SECRET`.
2. Set the identical secret in the terminal that starts ML, then run:

```powershell
$env:DEMO_MODE='false'
$env:INTERNAL_ML_SECRET='your-local-secret'
C:\venvs\dr\Scripts\python.exe -m uvicorn app:app --app-dir ml-service --reload --port 8000
```

3. In a second terminal, run `npm run dev`.

For an explicitly local-only, unauthenticated smoke environment, set `ALLOW_UNAUTHENTICATED_LOCAL_INFER=true`; never use that setting on Railway.

## Push to GitHub

Commit this repository, including `ml-service/Dockerfile` and the tracked ML artifacts, then push it to the existing EyeZen repository. Do not create a separate ML repository.

## Deploy Railway ML

1. In Railway, create a service from the current EyeZen GitHub repository.
2. Set **Root Directory** to `ml-service`.
3. Select the Dockerfile builder. Railway uses `ml-service/Dockerfile` from the repository (as `Dockerfile` within that root directory).
4. Add the environment variables below and deploy.
5. Generate a public domain for the service.
6. Confirm `https://<railway-domain>/health` returns HTTP 200 before configuring Vercel.

### Railway environment variables

```text
DEMO_MODE=false
INTERNAL_ML_SECRET=<long-random-value-shared-with-vercel>
```

Railway provides `PORT`; do not add it manually. Do not add `MONGODB_URI`, `JWT_SECRET`, or `BLOB_READ_WRITE_TOKEN`.

## Configure Vercel

1. Connect the same GitHub repository to Vercel and attach/create the EyeZen Vercel Blob store.
2. Add these server-only environment variables for Production (and Preview only when a corresponding Railway environment is available):

```text
MONGODB_URI=<Atlas URI>
JWT_SECRET=<long-random-value>
DEMO_MODE=false
STORAGE_MODE=vercel-blob
ML_SERVICE_URL=https://<railway-domain>
INTERNAL_ML_SECRET=<the-exact-Railway-secret>
CLIENT_ORIGIN=https://<production-vercel-domain>
BLOB_READ_WRITE_TOKEN=<provided by the connected Blob store>
```

3. Deploy Vercel. Its build should only install Node dependencies, build Vite, and create `api/index.js`; it must not detect or package Python/TensorFlow.

Remove obsolete `VERCEL_SUPPORT_LARGE_FUNCTIONS` and `VERCEL_ANALYZE_BUILD_OUTPUT` after the migration.

## Test the full application

1. Open `https://<railway-domain>/health` and verify real models report loaded.
2. Open the Vercel homepage, register/login, and submit a retinal image.
3. Verify grade, severity, confidence, five probabilities, referable probability, threshold, and decision.
4. Verify the U-Net overlay, Grad-CAM image, and PDF; reopen the record after logout/login.
5. Verify Name, Age, and Record ID filters, then use two accounts to confirm record and asset isolation.
6. Upload a JPG/PNG over 4.5 MB (up to 12 MB) and confirm it goes browser → Vercel Blob rather than multipart Express upload.

## Troubleshooting

- **Vercel packages Python:** check that `.vercelignore` contains `ml-service/**`, `api/ml.py` is absent, and there is no Python function/rewrite in `vercel.json`.
- **Railway 403 on inference:** `INTERNAL_ML_SECRET` differs between Railway and Vercel, or is missing.
- **Railway does not start:** inspect its logs for missing artifacts or insufficient memory; `/health` will not be available until models load.
- **Blob upload failure:** verify the Vercel Blob store/token and keep the service private-asset flow unchanged. Railway must not receive a Blob token.
- **Inference unavailable:** check Railway logs and `ML_SERVICE_URL`; Express intentionally returns a safe 502 rather than forwarding Railway internals.

## Redeploy procedure

1. Push the changes to the same GitHub repository.
2. Redeploy Railway first and wait for `/health` = 200.
3. Update Vercel's `ML_SERVICE_URL` only if the Railway domain changed, then redeploy Vercel.
4. Run the full-application test above.

## Deployment checklist

### RAILWAY

- [ ] GitHub repo connected
- [ ] Root directory `ml-service`
- [ ] Dockerfile detected
- [ ] `INTERNAL_ML_SECRET` added
- [ ] `DEMO_MODE=false`
- [ ] Public domain generated
- [ ] `/health` returns 200

### VERCEL

- [ ] Repo connected
- [ ] Blob store connected
- [ ] MongoDB URI configured
- [ ] JWT configured
- [ ] `STORAGE_MODE=vercel-blob`
- [ ] `ML_SERVICE_URL` is the Railway domain
- [ ] `INTERNAL_ML_SECRET` matches Railway
- [ ] `CLIENT_ORIGIN` configured
- [ ] Deployment passes without Python functions

### FINAL

- [ ] Login
- [ ] Upload
- [ ] Real V4
- [ ] U-Net
- [ ] Grad-CAM
- [ ] PDF
- [ ] History
- [ ] Filters
- [ ] Account isolation
