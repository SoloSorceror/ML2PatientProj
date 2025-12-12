from fastapi import FastAPI, HTTPException, UploadFile, File
import io
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from src.recommender_system import HealthRecommender

app = FastAPI(title="Patient Health Segmentation API", description="API for predicting patient clusters and generating health recommendations.")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for dev, or specifics like ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Load Models & Data ---
# Paths (Adjust if running from root e:\ML2PatientProj)
DATA_PATH = 'data/processed/umap_clustered_dataset.csv'
PREPROCESSOR_PATH = 'data/processed/preprocessor.pkl'
UMAP_PATH = 'data/processed/umap_reducer.pkl'
KMEANS_PATH = 'data/processed/kmeans_model.pkl'

try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    umap_reducer = joblib.load(UMAP_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    # Recommender needs to be initialized
    recommender = HealthRecommender(DATA_PATH, PREPROCESSOR_PATH, UMAP_PATH, KMEANS_PATH)
    print("Models and Recommender loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    raise e

# --- Pydantic Models ---
class PatientData(BaseModel):
    # Demographics
    Gender: float # 1.0 or 2.0 usually
    Age: float
    # Diet
    Calories: float
    Protein: float
    Carbs: float
    Sugar: float
    Fiber: float
    Fat: float
    # Exam
    BMI: float
    SystolicBP: float
    DiastolicBP: float
    # Labs
    Glucose: float
    TotalCholesterol: float
    # Other
    Diabetes: float # 0 or 1
    Smoker: float # 0 or 1
    
    # Missing columns that might be needed by preprocessor?
    # Based on data_processing.py, the columns were:
    # Gender, Age, Race, Education (Demographics)
    # Calories, Protein, Carbs, Sugar, Fiber, Fat (Diet)
    # BMI, SystolicBP, DiastolicBP (Exam)
    # Glucose, TotalCholesterol (Labs)
    # MedicationCount (Meds)
    # Diabetes, Smoker (Questionnaire)
    
    # We need all 19 features used in training.
    # Let's add the ones missing above.
    Race: float
    Education: float
    MedicationCount: float

class PredictionResponse(BaseModel):
    cluster: int
    cluster_name: str
    probabilities: list[float] = [] # Optional, if we want soft clustering

class RecommendationResponse(BaseModel):
    initial_cluster: int
    recommended_cluster: int # Target (2)
    actions: list[str]
    trajectory_plot_url: str = None # Could trigger a plot generation
    predicted_bmi_change: float
    predicted_glucose_change: float

class PredictionOut(BaseModel):
    initial_cluster: int
    actions: list[str]
    predicted_bmi_change: float
    predicted_glucose_change: float
    cluster: int
    cluster_name: str
    risk_profile: str
    narrative: str
    final_state: dict

# --- Cluster Names ---
CLUSTER_NAMES = {
    0: "Metabolic Hyper-Intake",
    1: "Pediatric Growth (Early)",
    2: "Optimal Biometric Profile",
    3: "Pediatric Growth (Late)",
    4: "Geriatric Cardiovascular Risk"
}

# --- Helper ---
def create_input_dataframe(data: PatientData):
    # Convert Pydantic model to DataFrame with correct column order
    # Order matters for the preprocessor!
    # We must match the order in `data_processing.py` -> `full_patient_dataset.csv`
    
    # Order from data_processing.py:
    # ['SEQN' (index), 'Gender', 'Age', 'Race', 'Education', 
    #  'Calories', 'Protein', 'Carbs', 'Sugar', 'Fiber', 'Fat', 
    #  'BMI', 'SystolicBP', 'DiastolicBP', 
    #  'Glucose', 'TotalCholesterol', 
    #  'MedicationCount', 
    #  'Diabetes', 'Smoker']
    
    # Note: SEQN is not a feature for the model usually, but preprocessor might handle it or ignore it.
    # The preprocessor likely expects just the feature columns if we trained with X = df.drop(columns=['SEQN'])
    # Checking segmentation_model.py... it used `df.drop(columns=['SEQN'])`
    
    row = {
        'Gender': data.Gender,
        'Age': data.Age,
        'Race': data.Race,
        'Education': data.Education,
        'Calories': data.Calories,
        'Protein': data.Protein,
        'Carbs': data.Carbs,
        'Sugar': data.Sugar,
        'Fiber': data.Fiber,
        'Fat': data.Fat,
        'BMI': data.BMI,
        'SystolicBP': data.SystolicBP,
        'DiastolicBP': data.DiastolicBP,
        'Glucose': data.Glucose,
        'TotalCholesterol': data.TotalCholesterol,
        'MedicationCount': data.MedicationCount,
        'Diabetes': data.Diabetes,
        'Smoker': data.Smoker
    }
    return pd.DataFrame([row])

# --- Endpoints ---

@app.get("/")
def home():
    return {"message": "Patient Health Segmentation API is running."}

from src.narrative_generator import MedicalNarrativeGenerator

@app.post("/predict_cluster", response_model=PredictionOut)
def predict_cluster(data: PatientData):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # 1. Preprocess
        processed_data = preprocessor.transform(input_df)
        
        # 2. UMAP
        umap_embedding = umap_reducer.transform(processed_data)
        
        # 3. KMeans
        cluster_id = kmeans.predict(umap_embedding)[0]
        initial_cluster = int(cluster_id)
        input_df['Cluster'] = initial_cluster
        
        cluster_name = CLUSTER_NAMES.get(initial_cluster, "Unknown")
        risk_profile = "High" if initial_cluster in [0, 4] else "Normal"

        # 4. Generate Recommendations
        rec_result = recommender.recommend(patient_row=input_df, max_steps=5)
        
        actions_taken = rec_result['actions']
        trajectory = rec_result['trajectory']
        current_state = rec_result['final_state']
        
        # Calculate Delta
        initial_bmi = input_df['BMI'].values[0]
        final_bmi = current_state['BMI'].values[0]
        initial_glucose = input_df['Glucose'].values[0]
        final_glucose = current_state['Glucose'].values[0]
        
        # Generate Narrative
        narrative = MedicalNarrativeGenerator.generate_narrative(
            patient_data=data.dict(),
            cluster_name=cluster_name,
            risk_profile=risk_profile,
            actions=actions_taken
        )
        
        return {
            "initial_cluster": initial_cluster,
            "cluster": initial_cluster, # Frontend uses this
            "cluster_name": cluster_name,
            "risk_profile": risk_profile,
            "narrative": narrative,
            "actions": actions_taken,
            "predicted_bmi_change": round(final_bmi - initial_bmi, 2),
            "predicted_glucose_change": round(final_glucose - initial_glucose, 2),
            "final_state": current_state.to_dict(orient='records')[0]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Ensure we have required columns, fill missing with defaults or error
        # Re-using the logic from data_processing might be cleaner but for now let's apply basics:
        
        # NOTE: In a production system we'd rigorously validate columns here.
        # For this prototype we assume headers match PatientData model or at least majority.
        
        # 1. Preprocess Batch
        # We need to ensure the columns match what preprocessor expects.
        # If headers are different, we might fail.
        
        # Impute missing if any (simple median strat or 0)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # We assume the preprocessor can handle this DF if it has correct columns
        # Filter to expected columns
        # expected_cols = ... (omitted for brevity, preprocessor usually handles strictness)
        
        processed_data = preprocessor.transform(df)
        
        # 2. UMAP
        # UMAP transform might be slow for huge batches, but fine for <10k rows
        umap_embedding = umap_reducer.transform(processed_data)
        
        # 3. KMeans
        cluster_ids = kmeans.predict(umap_embedding)
        
        results = []
        for i, cid in enumerate(cluster_ids):
            cname = CLUSTER_NAMES.get(int(cid), "Unknown")
            
            # Age override logic
            ag = df.iloc[i].get('Age', 0)
            if ag >= 18 and int(cid) in [1, 3]:
                cname = "Low Intake Adult / Outlier"
                
            results.append({
                "seqn": int(df.iloc[i].get('SEQN', i)), # Use Index if SEQN missing
                "cluster": int(cid),
                "cluster_name": cname,
                "risk_profile": "High" if int(cid) in [0, 4] else "Normal"
            })
            
        return results

    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Batch Processing Error: {str(e)}")
