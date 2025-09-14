from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os
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

def generar_prompt_explicacion(salario, clasificacion):
    prompt = (
        f"Soy un experto en análisis de salarios del mercado laboral Mid-Atlantic.\n"
        f"El salario anual informado es: {salario:.2f} mil dólares.\n"
        f"La clasificación calculada para este salario es: {clasificacion}.\n"
        "Explica al usuario qué significa esta clasificación en el contexto del dataset Wage (compara con la media, mediana y moda del salario) "
        "y ofrece 2 recomendaciones (económicas o profesionales) apropiadas para este nivel salarial."
    )
    return prompt

def resumen_salario(valores):
    # valores: lista de salarios, normalmente será solo [salario]
    salario = valores[0]
    if salario >= 170:
        clasificacion = 5
    elif salario >= 145:
        clasificacion = 4
    elif salario >= 120:
        clasificacion = 3
    elif salario >= 95:
        clasificacion = 2
    elif salario >= 70:
        clasificacion = 1
    else:
        clasificacion = 0
    return clasificacion

def explicar_clasificacion_salario(clasificacion: int):
    explicaciones = {
        0: "Muy bajo: riesgo de exclusión laboral/social.",
        1: "Bajo: remuneración insuficiente para nivel promedio.",
        2: "Medio: salario dentro del rango más común.",
        3: "Alto: salario por encima de la mediana del sector.",
        4: "Muy alto: remuneración destacada y superior a la mayoría.",
        5: "Excepcional: top salarial, muy por encima del promedio."
    }
    return explicaciones.get(clasificacion, "Clasificación desconocida")


@app.post("/predict")
def predict(data: WageInput):
    new_data = pd.DataFrame([data.dict()])
    pred = modelo.predict(new_data)[0]
    return {
        "prediccion_salario": round(float(pred), 2),
        "entrada": data.dict()
    }
