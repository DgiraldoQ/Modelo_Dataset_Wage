from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import google.generativeai as genai

# Configuración de API Key para Google Gemini
genai_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=genai_api_key)

# Cargar modelo entrenado
modelo = joblib.load("best_wage_model.pkl")

app = FastAPI(title="API Predicción de Salarios")

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de entrada para predicción
class WageInput(BaseModel):
    age: int = Field(..., example=30)
    education: str = Field(..., example="College Grad")
    jobclass: str = Field(..., example="Information")
    health: str = Field(..., example="Good")
    health_ins: str = Field(..., example="Yes")

    class Config:
        schema_extra = {
            "example": {
                "age": 30,
                "education": "College Grad",
                "jobclass": "Information",
                "health": "Good",
                "health_ins": "Yes"
            }
        }


@app.get("/")
def home():
    return {"mensaje": "API de Predicción de Salarios funcionando 🚀"}


@app.post("/predict")
def predict(data: WageInput):
    new_data = pd.DataFrame([data.dict()])
    pred = modelo.predict(new_data)[0]
    return {
        "prediccion_salario": round(float(pred), 2),
        "entrada": data.dict()
    }
