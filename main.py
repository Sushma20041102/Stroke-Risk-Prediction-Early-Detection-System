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

# Load trained model
model = joblib.load("stroke_xgb.pkl")

# SIMPLE, SAFE MANUAL ENCODING (prevents 500 error)
gender_map = {"Male": 1, "Female": 0}
yesno_map = {"Yes": 1, "No": 0}
work_map = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3}
res_map = {"Urban": 1, "Rural": 0}
smoke_map = {"never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3}

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

    # Convert everything to numbers BEFORE prediction
    encoded = {
        "age": data.age,
        "gender": gender_map.get(data.gender, 1),
        "hypertension": data.hypertension,
        "heart_disease": data.heart_disease,
        "ever_married": yesno_map.get(data.ever_married, 1),
        "work_type": work_map.get(data.work_type, 0),
        "Residence_type": res_map.get(data.Residence_type, 1),
        "avg_glucose_level": data.avg_glucose_level,
        "bmi": data.bmi,
        "smoking_status": smoke_map.get(data.smoking_status, 0)
    }

    df = pd.DataFrame([encoded])
    
    # Reorder columns to match training order (gender FIRST, then age)
    feature_order = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
                     'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
    df = df[feature_order]
    
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
