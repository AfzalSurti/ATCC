# Deploy ATCC — Vercel (frontend) + Render (API + video processor)

This project has **2 deployable pieces**:

1. **Frontend** → Vercel (`frontend/`)
2. **Backend + YOLO pipeline** → Render (`api/` + `src/`)

---

## Before you start

- GitHub repo: `https://github.com/AfzalSurti/ATCC`
- Accounts: [Render](https://render.com) + [Vercel](https://vercel.com)
- Expect **slow** processing on Render free/CPU (no GPU)

---

## STEP 1 — Deploy API on Render

1. Open https://dashboard.render.com → **New** → **Web Service**
2. Connect GitHub → select **AfzalSurti/ATCC**
3. Fill:

| Field | Value |
|--------|--------|
| Name | `atcc-api` |
| Language | Python 3 |
| Branch | `main` |
| Root Directory | *(leave empty)* |
| Build Command | see below |
| Start Command | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |

**Build Command** (copy all):

```bash
pip install --upgrade pip && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements-render.txt
```

4. Environment variables:

| Key | Value |
|-----|--------|
| `PYTHON_VERSION` | `3.11.9` |
| `CORS_ORIGINS` | `*` *(temporary; replace with Vercel URL in Step 3)* |
| `SERVE_FRONTEND` | `0` |

5. Click **Create Web Service** → wait for build (10–20+ min first time; torch is large).
6. When live, open:  
   `https://YOUR-SERVICE.onrender.com/api/health`  
   Expect: `{"status":"ok"}`

**Save your API URL** — you need it for Vercel.

---

## STEP 2 — Deploy frontend on Vercel

1. Open https://vercel.com → **Add New** → **Project**
2. Import **AfzalSurti/ATCC**
3. Configure:

| Field | Value |
|--------|--------|
| Root Directory | `frontend` |
| Framework Preset | Vite |
| Build Command | `npm run build` |
| Output Directory | `dist` |

4. Environment Variables:

| Key | Value |
|-----|--------|
| `VITE_API_BASE` | `https://YOUR-SERVICE.onrender.com` *(no trailing slash)* |

5. Deploy → open the Vercel URL (e.g. `https://atcc-xxx.vercel.app`)

---

## STEP 3 — Lock CORS to your Vercel URL

1. Render → your `atcc-api` → **Environment**
2. Set `CORS_ORIGINS` to your real Vercel URL, e.g.  
   `https://atcc-xxx.vercel.app`
3. **Manual Deploy** → clear build cache optional → redeploy
4. Hard-refresh the Vercel site and test upload

---

## STEP 4 — Smoke test

1. Open Vercel site  
2. Pick **2-way**  
3. Upload a **short** video (30–60 sec)  
4. Watch job progress  
5. Download Excel when done  

If the first request is slow: Render free tier **cold-starts** after idle.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Build fails / out of memory | Upgrade Render plan; torch is heavy |
| CORS error in browser | `CORS_ORIGINS` must match Vercel URL exactly |
| API 404 on `/api/...` | Confirm start command uses `api.main:app` |
| Upload works but never finishes | CPU too slow / service slept; use short video or paid plan |
| `VITE_API_BASE` ignored | Rebuild on Vercel after changing env vars |

---

## Optional: Blueprint deploy

Repo includes `render.yaml`. On Render: **New** → **Blueprint** → select this repo → set `CORS_ORIGINS` when prompted.
