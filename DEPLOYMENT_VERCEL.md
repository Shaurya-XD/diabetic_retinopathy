# EyeZen on Vercel

## Architecture

The Vite client is deployed as the static site. `/api/*` is a Node/Express function, and `/api/ml/*` is a Python/FastAPI function that imports the existing V4 service. MongoDB Atlas remains the database. In production, uploaded retinal images and generated overlays/PDFs must be stored in Vercel Blob; local development continues to use the local FastAPI output directory.

## Important ML limitation

The two model files are approximately 17 MB (`dr_multidomain_efficientnetb0_v4.keras`) and 70 MB (`idrid_lesion_unet_final.weights.h5`), 87 MB total before TensorFlow and OpenCV. Vercel's standard Python function bundle limit is 500 MB uncompressed. TensorFlow commonly pushes the final bundle beyond that limit. Enable **Fluid Compute** and **Large Functions (beta)** in the Vercel project before deploying the ML function; Large Functions supports up to 5 GB. Use Pro or Enterprise and select 4 GB / 2 vCPU in Settings -> Functions -> Advanced Settings for practical inference latency. Hobby is fixed at 2 GB / 1 vCPU and is not a reliable target for this workload.

## Required Vercel environment variables

Set these for Production and Preview; use local `.env` only for Development:

```text
MONGODB_URI=
JWT_SECRET=
DEMO_MODE=false
BLOB_READ_WRITE_TOKEN=
STORAGE_MODE=vercel-blob
ML_SERVICE_URL=
CLIENT_ORIGIN=https://your-domain.example
```

Never use `VITE_` for the MongoDB URI, JWT secret, ML URL, or Blob token. Create a Blob store in Storage and connect it to the project to obtain `BLOB_READ_WRITE_TOKEN`.

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

## Custom domain and cold starts

Add the domain in Project Settings -> Domains, then set `CLIENT_ORIGIN` to its HTTPS origin and redeploy. TensorFlow model initialization is expected on cold starts; model instances are cached at module startup per warm function instance.
