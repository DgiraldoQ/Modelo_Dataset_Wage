from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os
import google.generativeai as genai
from fastapi import HTTPException
from enum import Enum
import numpy as np


# Configuración de API Key para Google Gemini
genai_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=genai_api_key)

# Cargar modelo entrenado
modelo = joblib.load("best_wage_model.joblib")

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
class HealthEnum(str, Enum):
    regular_o_mala = "Regular o Mala"
    muy_buena_o_excelente = "Muy Buena o Excelente"

class WageInput(BaseModel):
    age: int
    education: str
    jobclass: str
    health: HealthEnum   # <- Aquí el cambio importante
    health_ins: str

    class Config:
        json_schema_extra = {
            "example": {
                "age": 30,
                "education": "College Grad",
                "jobclass": "Information",
                "health": "Muy Buena o Excelente",
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
    try:
        new_data = pd.DataFrame([data.dict()])
        new_data['logwage'] = np.nan
        print("Datos para predicción:", new_data)
        log_pred = modelo.predict(new_data)[0]
        print("Predicción logwage:", log_pred)
        wage_pred = np.exp(log_pred)
        print("Predicción wage:", wage_pred)
        clasif = resumen_salario([wage_pred])
        print("Clasificación:", clasif)
        explicacion = explicar_clasificacion_salario(clasif)
        prompt = generar_prompt_explicacion(wage_pred, clasif)
        return {
            "prediccion_salario": round(float(wage_pred), 2),
            "clasificacion": clasif,
            "explicacion": explicacion,
            "mensaje_explicativo": prompt,
            "entrada": data.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {e}")
