import streamlit as st
import pandas as pd
import requests
import os

st.set_page_config(page_title="Predicción de Salarios", layout="wide")

# Obtener la URL de la API
def get_api_url():
    try:
        return st.secrets["API_URL"]
    except Exception:
        return os.environ.get("API_URL", "http://127.0.0.1:8000")

API_URL = get_api_url().rstrip("/")

# Cargar dataset para visualizaciones
try:
    df = pd.read_csv("Wage.csv", sep=";")
except Exception:
    st.error("No se pudo cargar el archivo proyecto_normalizado.csv")
    st.stop()

st.title("💼 Dashboard - Predicción de Salarios")


# -------------------------------
# Sección: Predicción rápida
# -------------------------------
st.sidebar.header("Predicción rápida")

age = st.sidebar.number_input("Edad", min_value=18, max_value=70, value=30)
education = st.sidebar.selectbox("Educación", df["education"].unique().tolist())
jobclass = st.sidebar.selectbox("Trabajo", df["jobclass"].unique().tolist())
health = st.sidebar.selectbox("Salud", df["health"].unique().tolist())
health_ins = st.sidebar.selectbox("¿Seguro de salud?", df["health_ins"].unique().tolist())
maritl = st.sidebar.selectbox("Estado civil", df["maritl"].unique().tolist())
race = st.sidebar.selectbox("Raza", df["race"].unique().tolist())
year = st.sidebar.selectbox("Año", sorted(df["year"].unique().tolist()))

if st.sidebar.button("Predecir salario"):
    payload = {
        "age": age,
        "education": education,
        "jobclass": jobclass,
        "health": health,
        "health_ins": health_ins,
        "maritl": maritl,
        "race": race,
        "year": year
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        r.raise_for_status()
        data_resp = r.json()

          # Mostrar resultados completos
        st.sidebar.success(f"💰 Salario estimado: {data_resp.get('prediccion_salario')}")
        st.sidebar.write(f"📊 Clasificación salarial: {data_resp.get('clasificacion')}")
        st.sidebar.write(f"📝 Explicación: {data_resp.get('explicacion')}")
        st.sidebar.write("💡 Mensaje explicativo:")
        st.sidebar.text_area("", value=data_resp.get("mensaje_explicativo"), height=120)

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

# -------------------------------
# Sección: Información del proyecto
# -------------------------------
st.subheader("📖 Información del Proyecto")

st.markdown("""
**Título del estudio:** *Mid-Atlantic Wage Data | Análisis Estadístico*  
**Autor:** Diego Armando Giraldo Quintero  
**Fecha:** 17 de septiembre de 2025  

Este proyecto tiene como propósito analizar el dataset **Wage** del paquete ISLR, 
aplicando técnicas de estadística descriptiva y modelos de predicción.  
Se busca identificar cómo factores como **edad, educación, estado civil, raza, salud, 
ocupación y seguro médico** influyen en los salarios de los trabajadores 
de la región Mid-Atlantic (EE.UU.).  

# Datos iniciales:

## Eliminamos la columna región y utilizamos las variables restantes las cuales son las siguientes:

- wage: salario anual (en miles de dólares).

- year: año del registro (2003–2009).

- age: edad del trabajador.

- education: nivel educativo (factor con 5 niveles).

- race: grupo étnico.

- sex: género.

- jobclass: tipo de ocupación (industrial o de oficina).

- health: estado de salud (bueno / malo).

- health_ins: cobertura de seguro médico (sí / no).

# Medidas de tendencia central:

El dataset utilizado está compuesto por 3000 observaciones y 11 variables, de las cuales 7 son cualitativas (sexo, estado civil, raza, educación, clase de trabajo, estado de salud y cobertura de seguro médico) y 4 son cuantitativas (año, edad, salario y logaritmo del salario). Esta combinación permite realizar un análisis estadístico tanto descriptivo univariado como bivariado, 
explorando la relación entre características demográficas, laborales y los niveles salariales de los individuos.

## Dataset datos iniciales:

|index|Mean|Median|Mode|Variance|Standard Deviation|Range|IQR|Skewness|Kurtosis|
|---|---|---|---|---|---|---|---|---|---|
|year|2005\.791|2006\.0|2003\.0|4\.105354118039296|2\.026167346997601|6\.0|4\.0|0\.14288258983234128|-1\.2662568202657345|
|age|42\.41466666666667|42\.0|40\.0|133\.2271272646447|11\.54240560995171|62\.0|17\.25|0\.1478079016977663|-0\.446615538041673|
|wage|111\.70327|104\.92|118\.88|1741\.271120647327|41\.72854084014114|298\.25|43\.30000000000001|1\.6823543890847017|4\.839018179902112|

- Años (2003–2009): Los datos están equilibrados, sin concentrarse en un solo año.

- Edad (18–80): La mayoría de las personas tiene entre 33 y 50 años, con promedio de 42.

- Salarios: El salario típico ronda los 105–120, pero existen grandes diferencias: unos pocos muy altos elevan el promedio.

- Mensaje clave: La base refleja una fuerza laboral madura, bien distribuida en el tiempo, pero con desigualdad salarial marcada.

# Variables cualitativas:

## Variable educacion:

## Resultados:

- < HS Grad: 268 (8.9%)

- HS Grad: 971 (32.4%)

- Some College: 650 (21.7%)

- College Grad: 685 (22.8%)

- Advanced Degree: 426 (14.2%)

La mayoría de los individuos alcanzaron al menos la educación secundaria completa (HS Grad) (32.4%). Un porcentaje importante continuó estudios universitarios (College Grad 22.8% y Some College 21.7%). Solo un 8.9% no finalizó la secundaria. 
Esto muestra que la población estudiada tiene un nivel educativo relativamente alto, lo que puede influir directamente en los ingresos observados.

## Variable matril:

- Married: 2074 (69.1%)

- Never Married: 648 (21.6%)

- Divorced: 204 (6.8%)

- Separated: 55 (1.8%)

- Widowed: 19 (0.6%)

La gran mayoría de la población está casada (69.1%), seguida por quienes nunca se casaron (21.6%). 
Los demás estados civiles son poco frecuentes (menos del 10%). 
Esto sugiere que el dataset representa una población adulta en su mayoría establecida en matrimonios.

## Variable raza:

- White: 2480 (82.7%)

- Black: 293 (9.8%)

- Asian: 190 (6.3%)

- Other: 37 (1.2%)

La población es predominantemente White (82.7%), mientras que los demás grupos raciales representan minorías dentro del dataset. 
Esto puede limitar la capacidad de análisis comparativo entre razas, ya que la muestra no es balanceada.

## Variable jobclass

- Industrial: 1544 (51.5%)

- Information: 1456 (48.5%)

La distribución es bastante equilibrada entre quienes trabajan en el sector Industrial (51.5%) y el sector Information (48.5%). 
Esto permite analizar comparaciones directas entre ambos grupos laborales con buena representatividad.

## Variable Health;

- Muy buena o excelente: 2142 (71.4%)

- Regular o mala: 858 (28.6%)

La mayoría de los individuos se perciben con una salud buena o excelente (71.4%), mientras que poco más de una cuarta parte reporta tener salud regular o mala (28.6%). 
Esto refleja una percepción positiva del estado de salud en general.

## Variable Health_ins;

- Yes: 2083 (69.4%)

- No: 917 (30.6%)

Alrededor de 7 de cada 10 personas cuentan con seguro de salud (69.4%), mientras que un 30.6% no tiene cobertura. 
Esto indica un nivel relativamente alto de acceso a servicios de salud dentro de la población analizada.

## Nivel educativo vs salarios

El análisis del salario promedio según nivel educativo muestra que los ingresos aumentan de manera general con la educación. 
El grupo con educación incompleta universitaria (“Some College”) presenta el salario promedio más alto (171.27), 
superando incluso a graduados universitarios y con posgrado. Esto sugiere que factores adicionales, como la experiencia laboral o el sector de ocupación, influyen de forma significativa en el nivel salarial. En contraste, 
los individuos con menor nivel educativo (“< HS Grad”) obtienen los salarios más bajos (84.10), evidenciando la brecha económica vinculada a la educación.

|index|0\.25|0\.5|0\.75|
|---|---|---|---|
|age|33\.75|42\.0|51\.0|
|wage|85\.38|104\.92|128\.68|

# Análisis de Salario

## Varianza:

La varianza obtenida para la variable wage es de 363,248.8, lo que indica una alta dispersión en los ingresos de los individuos. 
Dado que la varianza se mide en unidades cuadráticas, su interpretación directa no es intuitiva.

## Desviacion estandar:

La desviación estándar de los salarios es 602.70, lo que implica que los ingresos de los individuos presentan una gran dispersión respecto al promedio. Dado que este valor es varias veces superior a la media de los salarios (~111), 
se concluye que la distribución salarial es altamente heterogénea, con predominio de valores bajos y la presencia de algunos ingresos extremadamente elevados que incrementan significativamente la variabilidad.

## Coeficiente de desviacion

El coeficiente de variación calculado para la variable wage es de 430.75%, lo que refleja una dispersión extremadamente alta respecto al salario promedio. 
Esto significa que la variabilidad de los ingresos es más de cuatro veces superior al valor medio, evidenciando una marcada desigualdad salarial. 
En consecuencia, la media no resulta un indicador representativo de los salarios, siendo la mediana una medida más adecuada para describir la tendencia central de los datos.

## Quantiles

Los quantiles muestran la distribución de los salarios en distintos puntos de corte. El 1% más bajo percibe salarios menores a 40.41, 
mientras que aproximadamente un tercio de la población obtiene ingresos inferiores a 90. Entre los percentiles 27 y 32 la variación es mínima, 
lo que evidencia una acumulación de observaciones en ese rango. La mediana, ubicada en 104.92, indica que el 50% de la población gana menos de este valor y el 50% restante más, lo que la convierte en una medida central más representativa que la media, dada la fuerte dispersión observada en los salarios.


""")

st.markdown("## Quantiles")
st.image("img/Quantiles.png", caption="Quantiles", width=1000)



st.markdown("""

# ¿Por qué limpiar el dataset wage?

- Valores extremos (outliers)

- En la variable salario (wage) se observan sueldos muy altos comparados con la mayoría.

- Estos valores distorsionan el promedio y pueden llevar a conclusiones equivocadas.
👉 Ejemplo: la media es mayor a la mediana, lo que indica que los salarios muy altos están inflando el promedio.

- Rango amplio de edades

- Hay personas desde 18 hasta 80 años.

- Algunas edades pueden no ser representativas para el análisis laboral (ej. casos atípicos en edades muy altas).

- Distribuciones sesgadas

- El salario presenta sesgo positivo (skewness > 1.5) → hay muchos sueldos bajos y pocos muy altos.

- Esto afecta modelos estadísticos o de machine learning, que suelen asumir distribuciones más balanceadas.

- Posibles datos faltantes o inconsistentes

- Antes del análisis, es necesario revisar si hay NA o valores nulos en variables clave (edad, salario, educación, etc.).

- También verificar registros con datos duplicados o fuera de rango.


Tras la limpieza, el dataset wage muestra una distribución salarial más representativa: la media y la mediana son muy cercanas (≈106 vs. 104), 
lo que elimina el sesgo que provocaban valores atípicos. 
Esto hace que cualquier análisis o modelo construido sobre estos datos sea más robusto y fiel a la realidad laboral.

|index|year|age|wage|
|---|---|---|---|
|count|2881\.0|2881\.0|2881\.0|
|mean|2005\.7889621659147|42\.2707393266227|106\.2170357514752|
|std|2\.02687575991661|11\.5987730332487|30\.596953433383234|
|min|2003\.0|18\.0|20\.93|
|25%|2004\.0|33\.0|84\.05|
|50%|2006\.0|42\.0|103\.9|
|75%|2008\.0|50\.0|127\.12|
|max|2009\.0|80\.0|193\.45|

# Despues de la limpieza realizamos la comparacion

La comparación de medidas de tendencia central muestra el impacto de los valores atípicos en la distribución salarial. 
La media original (139.87) estaba claramente distorsionada, mientras que tras la limpieza se alinea con la mediana (106.22 vs. 103.90), reflejando una distribución más simétrica y representativa. La mediana, por su robustez, apenas se vio afectada, y la moda permaneció constante en 40. 
Esto confirma que la depuración de datos permitió obtener un panorama mucho más realista del comportamiento salarial.

| Medida  | Original | Limpio  |
|---------|----------|---------|
| Media   | 139.8698 | 106.217 |
| Mediana | 104.9200 | 103.900 |
| Moda    | 118.8800 | 118.880 |

# Visualizar antes y después (boxplots o histogramas)

""")

st.markdown("## boxplot salario original vs salario limpio")
st.image("img/salariooriginal_vs_salariolimpio.png", caption="Quantiles", width=1000)

st.markdown("""
La comparación entre ambos histogramas muestra que, en el dataset original, la presencia de valores atípicos (outliers) 
generaba una escala distorsionada que ocultaba la distribución de la mayoría de los salarios. Tras la limpieza de datos, se observa una distribución clara, 
con la mayor parte de los salarios ubicados entre 50 y 200, lo que permite un análisis estadístico más representativo y confiable.

""")

st.markdown("## histograma salario original vs salario limpio")
st.image("img/histograma.png", caption="Quantiles", width=1000)

st.markdown("""
# Histograma Original (izquierda)

- La mayoría de los datos aparecen concentrados cerca de 0, con un gran bloque de frecuencias.

- Se observan valores atípicos muy altos (outliers) que llegan hasta más de 12,000.

- Estos valores extremos distorsionan la distribución, haciendo que no se aprecie la forma real de los datos.

- Esto explica por qué la media (139.86) es mucho mayor que la mediana (104.92): la media está inflada por esos outliers.

#Histograma Limpio (derecha)

- Después de eliminar o corregir los valores atípicos, la distribución se ve mucho más clara.

- Ahora los datos siguen una forma aproximadamente normal (campana), concentrándose entre 50 y 150.

- Se observa un rango más realista para salarios y una dispersión más representativa de la población.

- La media (106.21) y la mediana (103.9) ahora están mucho más cercanas, lo que indica menos sesgo en los datos.

""")

st.markdown("## histograma salario original vs salario limpio")
st.image("img/distribucion_segun_raza.png", caption="Quantiles", width=1000)
