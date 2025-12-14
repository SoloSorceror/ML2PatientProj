# 🏥 Cortex Med: AI Patient Segmentation System
> *Next-generation computational patient analysis & neural health segmentation engine.*

## 🚀 Overview
**Cortex Med** is an advanced health analytics platform that uses unsupervised machine learning (UMAP + KMeans) to segment patients into distinct metabolic profiles. It provides personalized, actionable health optimization plans using a custom Reinforcement Learning inspired recommendation engine.

![Dashboard Preview](frontend/public/dashboard_preview.png)

---

## ✨ Features (V2.0)
- **🧠 Neural Segmentation**: Categorizes patients into 5 distinct clusters (e.g., "Healthy Young Adult", "Middle-Aged Metabolic Risk") using 19+ biometric markers.
- **⚡ Real-Time Diagnostics**: Instant analysis of patient risk profiles via FastAPI backend.
- **🔮 Trajectory Simulation**: Predicts health outcomes (BMI, Glucose reduction) based on suggested interventions.
- **📂 Batch Analysis**: Process hundreds of patient records via CSV upload ("Batch Mode").
- **🕸️ Vitals Footprint**: Interactive Radar Chart visualizing multidimensional health metrics.
- **💎 Premium UI**: "Clinical SaaS" design system with glassmorphism, dark mode, and fluid animations.

---

## 🏗️ Tech Stack
| Component | Technology |
|:---|:---|
| **Backend** | **FastAPI** (Python), Uvicorn, Pandas, Scikit-Learn |
| **Frontend** | **React** (Vite), TailwindCSS, Framer Motion, Recharts |
| **ML Engine** | UMAP (Dimensionality Reduction), KMeans (Clustering) |
| **Data** | NHANES (CDC) Dataset |

---

## 🛠️ Installation & Setup

### 1. Backend Setup
Navigate to the root directory:
```bash
# Install dependencies
pip install -r requirements.txt

# Start the API Server
python -m src.api
# OR
uvicorn src.api:app --reload
```
*Server will run at `http://127.0.0.1:8000`*

### 2. Frontend Setup
Open a new terminal in `frontend/`:
```bash
cd frontend
npm install
npm run dev
```
*Application will run at `http://localhost:5173`*

---

## 📂 Project Structure

```bash
e:\ML2PatientProj\
├── data/
│   ├── processed/         # Serialized Models (.pkl) & Datasets
├── frontend/              # React Application
│   ├── src/
│       ├── pages/         # Dashboard, Home, BatchUpload
│       └── ...
├── src/
│   ├── api.py             # FastAPI App Entry Point
│   ├── recommender_system.py # Recommendation Logic
└── ...
```

## 🧪 Usage
1.  **Initialize System**: Click "Initialize System" on the landing page.
2.  **Dashboard**: Enter patient vitals (Age, BMI, Glucose, etc.).
3.  **Analysis**: View the predicted Cluster and "Neural Optimization Plan".
4.  **Batch Mode**: Go to `/batch` to upload `test_batch.csv` for bulk analysis.

---
*Developed for Advanced Agentic Coding Project.*
