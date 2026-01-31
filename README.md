

```markdown
# Sentiment Analysis MLOps Pipeline with Fine-Tuned BERT

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

## 📌 Project Overview
This project is a production-grade MLOps system that performs **Sentiment Analysis** (Positive/Negative) using a **DistilBERT** model fine-tuned on the IMDb dataset. 

It demonstrates a complete end-to-end pipeline:
1.  **Data Engineering:** Automated preprocessing and class balancing to prevent model bias.
2.  **Model Training:** Fine-tuning a Transformer model to achieve **~90% accuracy**.
3.  **Deployment:** Serving the model via a **FastAPI** REST endpoint and a **Streamlit** UI.
4.  **Containerization:** Fully Dockerized architecture orchestrated with Docker Compose.

---

## 📂 Project Structure
```bash
├── data/
│   ├── processed/       # Balanced train/test CSV files
├── model_output/        # Fine-tuned model artifacts (SafeTensors, Config, Vocab)
├── results/             # Evaluation metrics and batch prediction outputs
├── scripts/
│   ├── preprocess.py    # Data cleaning and balancing strategy
│   ├── train.py         # Training loop with Hugging Face Trainer
│   └── batch_predict.py # Bulk inference script
├── src/
│   ├── api.py           # FastAPI backend with /predict and /health
│   └── ui.py            # Streamlit dashboard
├── .env.example         # Environment variable template
├── docker-compose.yml   # Multi-container orchestration
├── Dockerfile.api       # Backend container spec
├── Dockerfile.ui        # Frontend container spec
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

```

---

## 🛠️ Setup & Installation

### Prerequisites

* **Docker Desktop** (Running)
* **Python 3.11+** (For local execution)
* **Git**

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Twitter_Sentiment_Analysis_API_with_BERT_Fine-Tuning

```

### 2. Environment Configuration

Create a `.env` file from the example template.

```bash
cp .env.example .env

```

---

## 🚀 Running with Docker (Recommended)

The entire application is containerized. You can launch the API and UI with a single command.

**1. Build and Start Services:**

```bash
docker-compose up --build

```

**2. Access the Application:**

* **Web Interface (UI):** [http://localhost:8501](https://www.google.com/search?q=http://localhost:8501)
* **API Documentation:** [http://localhost:8000/docs](https://www.google.com/search?q=http://localhost:8000/docs)
* **API Health Check:** [http://localhost:8000/health](https://www.google.com/search?q=http://localhost:8000/health)

**3. Stop Services:**

```bash
docker-compose down

```

---

## 💻 Local Development & Training

If you want to retrain the model or run scripts locally:

### 1. Install Dependencies

```bash
pip install -r requirements.txt

```

### 2. Preprocess Data

Download and balance the IMDb dataset (handles Class Imbalance correction).

```bash
python scripts/preprocess.py

```

*Creates `data/processed/train.csv` and `data/processed/test.csv`.*

### 3. Train the Model

Fine-tune DistilBERT on the processed data.

```bash
python scripts/train.py

```

* Saves model artifacts to `model_output/`.
* Logs metrics to `results/metrics.json`.
* Logs hyperparameters to `results/run_summary.json`.

**Note:** The `model_output/` folder is required for Docker containers to build successfully.

---

## 📊 Model Performance

The model was fine-tuned for 3 epochs using a balanced dataset to resolve initial "Class Collapse" issues.

**Final Test Metrics (`results/metrics.json`):**
| Metric | Score |
| :--- | :--- |
| **Accuracy** | **89.70%** |
| **F1 Score** | **0.8985** |
| **Precision** | **0.8854** |
| **Recall** | **0.9120** |

---

## 🔌 API Reference

### Health Check

* **Endpoint:** `GET /health`
* **Response:** `{"status": "ok"}`

### Predict Sentiment

* **Endpoint:** `POST /predict`
* **Request Body:**
```json
{
  "text": "The implementation of this project is excellent."
}

```


* **Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.985
}

```



---

## 📁 Batch Predictions

To generate predictions for a large dataset (CSV file), run the batch script:

```bash
python scripts/batch_predict.py --input data/processed/test.csv --output results/predictions.csv

```

* **Input:** CSV with a `text` column.
* **Output:** CSV with `text`, `predicted_sentiment`, and `confidence` columns.

---

## 👨‍💻 Author

**Sarvesh**
*3rd Year B.Tech, Artificial Intelligence & Machine Learning*

