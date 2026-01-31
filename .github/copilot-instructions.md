# Copilot instructions for this repository

Purpose: Give an AI coding agent the minimal, actionable knowledge to be immediately productive working in this repo.

- Project layout: top-level scripts in `scripts/`, model artifacts in `model_output/` (runtime) and `model.safetensors` at repo root (checked-in). The API is in `src/api.py`; the (Streamlit) UI entry is `src/ui.py` (currently empty).

- Big picture:
  - `scripts/train.py` trains a DistilBERT classifier and saves tokenizer + model to `model_output/` and writes metrics to `results/`.
  - `src/api.py` loads a model from `./model_output` at import time and exposes two endpoints: `GET /health` and `POST /predict`.
  - `docker-compose.yml` defines two services, `api` and `ui`; the API image uses `Dockerfile.api` and maps host `./model_output` into `/app/model_output` inside the container.

- Important patterns & gotchas (do not change without tests):
  - The API hardcodes `MODEL_PATH = "./model_output"` in `src/api.py` and does not read the `MODEL_PATH` environment variable set in `docker-compose.yml`. If you change the container mapping or want env-driven paths, update `src/api.py` accordingly and add a test.
  - Label mapping is explicit in `src/api.py`: index `0` -> `negative`, index `1` -> `positive`. Use this mapping when creating tests or UI components.
  - `requirements.txt` pins `torch` using the CPU wheels index. Container builds rely on that URL to install CPU PyTorch.
  - `scripts/train.py` uses the Hugging Face `Trainer` API and saves artifacts to `model_output/`; training is intentionally short and meant as an example. It sets `eval_strategy`/`save_strategy` to `"no"` and writes placeholder metrics into `results/`.

- Developer workflows (concrete commands):
  - Run a quick smoke dataset setup: `python scripts/test_setup.py` (creates `data/processed/train.csv`).
  - Train locally (small example): `python scripts/train.py`.
  - Run the API locally: `uvicorn src.api:app --reload --host 0.0.0.0 --port 8000`.
  - Call the API (example):

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"text":"I love this!"}'
```

  - Run the stack with Docker Compose: `docker-compose up --build` (this maps local `model_output/` into the `api` container at `/app/model_output`).

- What to look for when modifying code:
  - Keep API route names and response schema stable (`/predict` returns `{"sentiment":..., "confidence":...}`). Changing these requires updating the UI and any integration tests.
  - Because the API imports the model at top-level, edits to `src/api.py` that affect model loading will cause slower reloads and may require restarting the process.
  - When changing training hyperparameters, keep `model_output/` layout compatible with `transformers` `save_pretrained()` and `from_pretrained()` (tokenizer and model files alongside `config.json` and tokenizer files).

- Integration points & external dependencies:
  - Hugging Face `transformers` and `datasets` (used by `scripts/train.py`).
  - PyTorch; `requirements.txt` uses an explicit wheel index for CPU builds.
  - Docker / Docker Compose for running the API + UI together; `docker-compose.yml` wires the `api` and `ui` services and sets `API_URL` for the UI.

- Quick examples to copy into tests or PR descriptions:
  - Model load smoke test (pseudo): import `AutoTokenizer, AutoModelForSequenceClassification` and assert `AutoTokenizer.from_pretrained('model_output')` succeeds when `model_output/` exists.
  - API contract test: POST `{"text":"test"}` to `/predict` and assert response has keys `sentiment` and `confidence`, and `sentiment` in `["negative","positive"]`.

- Notes for contributors / AI agents:
  - If `src/ui.py` is updated to call the API, read `docker-compose.yml` to ensure `API_URL` matches (`http://api:8000` inside compose; `http://localhost:8000` for local runs).
  - Prefer changing `scripts/*` behavior over editing `src/api.py` unless you are explicitly changing API functionality. The API is the integration contract for clients.

If anything in this file is unclear or you want more examples (unit tests, CI steps, or a sample UI implementation), tell me which area to expand and I will iterate.
