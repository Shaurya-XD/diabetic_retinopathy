# EyeZen on Vercel

## Architecture

The Vite client is deployed as the static site. `/api/*` is a Node/Express function, and `/api/ml/*` is a Python/FastAPI function that imports the existing V4 service. MongoDB Atlas remains the database. In production, uploaded retinal images and generated overlays/PDFs must be stored in Vercel Blob; local development continues to use the local FastAPI output directory.

## Important ML limitation

The two model files are approximately 17 MB (`dr_multidomain_efficientnetb0_v4.keras`) and 70 MB (`idrid_lesion_unet_final.weights.h5`), 87 MB total before TensorFlow and OpenCV. The current ML function bundles at approximately 2.08 GB uncompressed, above Vercel's standard 500 MB Python limit. It requires **Large Functions (public beta)**, which permits up to 5 GB on **Fluid Compute with Active CPU enabled**. New projects are enrolled automatically; for existing projects add `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` and redeploy. Set `VERCEL_ANALYZE_BUILD_OUTPUT=1` on a diagnostic deployment to inspect bundle contributors. For practical TensorFlow inference, use Pro/Enterprise and select Performance (4 GB / 2 vCPU) in Settings -> Functions -> Advanced Settings. Do not enable Secure Compute or Static IPs for this function, as Large Functions does not support them.

## Required Vercel environment variables

Set these for Production and Preview; use local `.env` only for Development:

```text
MONGODB_URI=
JWT_SECRET=
DEMO_MODE=false
BLOB_READ_WRITE_TOKEN=
STORAGE_MODE=vercel-blob
ML_SERVICE_URL= # optional on Vercel; set this only for an external ML service
CLIENT_ORIGIN=https://your-domain.example
VERCEL_SUPPORT_LARGE_FUNCTIONS=1 # required for existing projects; omit if automatically enrolled
VERCEL_ANALYZE_BUILD_OUTPUT=1 # optional one-deployment bundle analysis
```

Never use `VITE_` for the MongoDB URI, JWT secret, ML URL, or Blob token. Create a Blob store in Storage and connect it to the project to obtain `BLOB_READ_WRITE_TOKEN`. In Vercel, enable **Settings -> Environment Variables -> Automatically expose System Environment Variables** so `VERCEL_URL` is available; when `ML_SERVICE_URL` is omitted, the API derives `https://$VERCEL_URL/api/ml`.

## Local commands

```powershell
$env:DEMO_MODE='false'
C:\venvs\dr\Scripts\python.exe -m uvicorn app:app --app-dir ml-service --reload --port 8000
npm run dev
```

## Deploy

```powershell
npx vercel login
npx vercel link
npx vercel build
npx vercel deploy
npx vercel deploy --prod
```

After deployment, test `/api/ml/health`, login, upload, inference, stored Blob URLs, PDF download, history filters, and cross-account access. Use Deployments -> Logs and Functions -> Observability for failures. Roll back from Deployments by promoting the previous deployment.

## Large Functions dashboard steps

1. Open the EyeZen project, then Settings -> Functions, and confirm Fluid Compute / Active CPU is enabled (it is default for new projects).
2. For an existing project, add `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` in Settings -> Environment Variables for Preview and Production, then redeploy. New projects created after 30 June 2026 are enrolled automatically.
3. Add `VERCEL_ANALYZE_BUILD_OUTPUT=1` for a diagnostic redeploy, inspect the function-size report, then remove it if it is no longer needed.
4. On Pro/Enterprise, choose Performance (4 GB / 2 vCPU) in Settings -> Functions -> Advanced Settings. Hobby is fixed at Standard (2 GB / 1 vCPU).

## Custom domain and cold starts

Add the domain in Project Settings -> Domains, then set `CLIENT_ORIGIN` to its HTTPS origin and redeploy. TensorFlow model initialization is expected on cold starts; model instances are cached at module startup per warm function instance.
