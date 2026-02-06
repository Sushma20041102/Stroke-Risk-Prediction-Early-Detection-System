from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Stroke Risk Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model and encoders
model = joblib.load("stroke_xgb.pkl")
encoders = joblib.load("encoders.pkl")

# Create reverse mappings (category_name -> code)
def create_reverse_map(encoder_dict):
    return {v: k for k, v in encoder_dict.items()}

# Build encoding maps from the saved encoders
gender_map = create_reverse_map(encoders["gender"])
yesno_map = create_reverse_map(encoders["ever_married"])
work_map = create_reverse_map(encoders["work_type"])
res_map = create_reverse_map(encoders["Residence_type"])
smoke_map = create_reverse_map(encoders["smoking_status"])

class StrokeInput(BaseModel):
    age: int
    gender: str
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

@app.get("/")
def home():
    return {"message": "Stroke API running 🚀"}

@app.post("/predict")
def predict(data: StrokeInput):

    # Create DataFrame with correct column order and feature names
    # Order must match training: gender FIRST, then age
    feature_order = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
                     'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
    
    # Build data in correct order with proper encoding
    encoded_data = [
        gender_map.get(data.gender, 1),
        data.age,
        data.hypertension,
        data.heart_disease,
        yesno_map.get(data.ever_married, 1),
        work_map.get(data.work_type, 0),
        res_map.get(data.Residence_type, 1),
        data.avg_glucose_level,
        data.bmi,
        smoke_map.get(data.smoking_status, 0)
    ]
    
    # Create DataFrame with explicit column names
    df = pd.DataFrame([encoded_data], columns=feature_order)
    
    try:
        pred = int(model.predict(df)[0])
        return {
            "stroke_risk": pred,
            "risk_level": "High" if pred == 1 else "Low"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
