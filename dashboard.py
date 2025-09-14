import streamlit as st
import pandas as pd
import requests
import os

st.set_page_config(page_title="Predicción de Salarios", layout="wide")

# Obtener la URL de la API
API_URL = "https://modelo-dataset-wage.onrender.com"


st.title("💼 Dashboard - Predicción de Salarios")

# -------------------------------
# Sección: Predicción rápida
# -------------------------------
st.sidebar.header("Predicción rápida")

age = st.sidebar.number_input("Edad", min_value=18, max_value=70, value=30)
education = st.sidebar.selectbox("Educación", ["< HS Grad", "HS Grad", "Some College", "College Grad"])
jobclass = st.sidebar.selectbox("Trabajo", ["Industrial", "Information"])
health = st.sidebar.selectbox("Salud", ["Regular o Mala", "Good", "Muy Buena o Excelente"])
health_ins = st.sidebar.selectbox("¿Seguro de salud?", ["Yes", "No"])

if st.sidebar.button("Predecir salario con API"):
    payload = {
        "age": age,
        "education": education,
        "jobclass": jobclass,
        "health": health,
        "health_ins": health_ins
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        r.raise_for_status()
        data_resp = r.json()

        st.sidebar.success(f"💰 Salario estimado: {data_resp.get('prediccion_salario')}")
        st.sidebar.write("**Entrada enviada:**")
        st.sidebar.json(data_resp.get("entrada"))
    except Exception as e:
        st.sidebar.error(f"Error al conectar con la API: {e}")

# -------------------------------
# Sección: Vista previa dataset
# -------------------------------
st.subheader("📋 Vista previa de datos")
try:
    df = pd.read_csv("Wage.csv", sep=";")
    st.dataframe(df.head())
except Exception:
    st.warning("No se pudo cargar el dataset `Wage.csv`. Verifica que esté en el repositorio.")
