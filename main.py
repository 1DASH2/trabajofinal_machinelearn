import pandas as pd

# Cargar datos financieros
ruta = "datos/arch_financiero.csv"

df = pd.read_csv(ruta)

print("Datos cargados correctamente.")
print(f"Cantidad de registros: {len(df)}")

# Renombrar columna de fecha
df = df.rename(columns={
    "Unnamed: 0": "Date"
})

# Convertir fecha
df["Date"] = pd.to_datetime(df["Date"])

# Eliminar duplicados
df = df.drop_duplicates()

# Comprobar valores faltantes
print("\nDatos faltantes:")
print(df.isnull().sum())

# Ordenar por fecha
df = df.sort_values("Date")

# ==========================================
# 7. CREAR VARIABLES FINANCIERAS
# ==========================================

# Cambio porcentual del precio de cierre
df["Cambio_Porcentual"] = df["close"].pct_change() * 100

# Media móvil de 7 días
df["Media_Movil_7"] = df["close"].rolling(window=7).mean()

# Media móvil de 30 días
df["Media_Movil_30"] = df["close"].rolling(window=30).mean()

# Volatilidad de los últimos 7 días
df["Volatilidad_7"] = df["Cambio_Porcentual"].rolling(window=7).std()


# ==========================================
# 8. CREAR VARIABLE OBJETIVO
# ==========================================

# Precio de cierre del siguiente día
df["Close_Siguiente"] = df["close"].shift(-1)

# 1 = el precio sube
# 0 = el precio baja o se mantiene
df["Objetivo"] = (
    df["Close_Siguiente"] > df["close"]
).astype(int)


# ==========================================
# 9. ELIMINAR FILAS SIN INFORMACIÓN
# ==========================================

df = df.dropna()


# ==========================================
# 10. MOSTRAR RESULTADOS
# ==========================================

print("\nNuevas variables creadas:")
print(df[
    [
        "Date",
        "close",
        "Cambio_Porcentual",
        "Media_Movil_7",
        "Media_Movil_30",
        "Volatilidad_7",
        "Objetivo"
    ]
].head(10))

print("\nDistribución de la variable objetivo:")
print(df["Objetivo"].value_counts())

# Mostrar información
print("\nInformación del dataset:")
df.info()

print("\nPrimeras filas:")
print(df.head())

# ==========================================
# 11. VISUALIZACIÓN DE DATOS
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# GRÁFICO 1: EVOLUCIÓN DEL PRECIO
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["Date"],
    df["close"],
    label="Precio de cierre"
)

plt.plot(
    df["Date"],
    df["Media_Movil_7"],
    label="Media móvil 7 días"
)

plt.plot(
    df["Date"],
    df["Media_Movil_30"],
    label="Media móvil 30 días"
)

plt.title("Evolución del precio de cierre")
plt.xlabel("Fecha")
plt.ylabel("Precio")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("visualizaciones/evolucion_precio.png")

plt.show()


# ==========================================
# GRÁFICO 2: HISTOGRAMA
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["close"],
    bins=30,
    kde=True
)

plt.title("Distribución de los precios de cierre")
plt.xlabel("Precio de cierre")
plt.ylabel("Frecuencia")
plt.tight_layout()

plt.savefig("visualizaciones/histograma_precios.png")

plt.show()


# ==========================================
# GRÁFICO 3: DISPERSIÓN
# ==========================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="volume",
    y="close"
)

plt.title("Relación entre volumen y precio de cierre")
plt.xlabel("Volumen")
plt.ylabel("Precio de cierre")
plt.tight_layout()

plt.savefig("visualizaciones/dispersion_volumen_precio.png")

plt.show()


# ==========================================
# GRÁFICO 4: DISTRIBUCIÓN DE LA TENDENCIA
# ==========================================

plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="Objetivo"
)

plt.title("Distribución de tendencias del mercado")
plt.xlabel("Tendencia")
plt.ylabel("Cantidad")

plt.xticks(
    [0, 1],
    ["Baja / Igual", "Sube"]
)

plt.tight_layout()

plt.savefig("visualizaciones/distribucion_tendencia.png")

plt.show()

# ==========================================
# MACHINE LEARNING - CLASIFICACIÓN                   //(se le dice al programa "utiliza estas características del mercado para intentar predecir Objetivo")
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Variables que utilizaremos para hacer la predicción
caracteristicas = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "Cambio_Porcentual",
    "Media_Movil_7",
    "Media_Movil_30",
    "Volatilidad_7"
]

X = df[caracteristicas]
y = df["Objetivo"]

print("\nDatos preparados para Machine Learning.")
print("Cantidad de datos:", len(X))

# Separar los datos en entrenamiento y prueba
# 80% para entrenar y 20% para evaluar

tamano_entrenamiento = int(len(X) * 0.8)

X_train = X.iloc[:tamano_entrenamiento]
X_test = X.iloc[tamano_entrenamiento:]

y_train = y.iloc[:tamano_entrenamiento]
y_test = y.iloc[tamano_entrenamiento:]

print("\nDivisión de los datos:")
print("Datos de entrenamiento:", len(X_train))
print("Datos de prueba:", len(X_test))

# Escalar los datos
escalador = StandardScaler()

X_train_escalado = escalador.fit_transform(X_train)
X_test_escalado = escalador.transform(X_test)

# Crear el modelo
modelo = LogisticRegression(max_iter=1000)

# Entrenar el modelo
modelo.fit(X_train_escalado, y_train)

print("\nModelo de Machine Learning entrenado correctamente.")

# Realizar predicciones
predicciones = modelo.predict(X_test_escalado)

print("\nPredicciones realizadas correctamente.")
print("Primeras 10 predicciones:")
print(predicciones[:10])

# Calcular la precisión
precision = accuracy_score(y_test, predicciones)

print("\n==========================================")
print("RESULTADOS DEL MODELO")
print("==========================================")
print(f"Precisión del modelo: {precision:.2%}")

print("\nReporte de clasificación:")
print(classification_report(y_test, predicciones))

# ==========================================
# MACHINE LEARNING - CLASIFICACIÓN
# ==========================================

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Variables que utilizaremos para la predicción
caracteristicas = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "Cambio_Porcentual",
    "Media_Movil_7",
    "Media_Movil_30",
    "Volatilidad_7"
]

X = df[caracteristicas]
y = df["Objetivo"]

print("\nDatos preparados para Machine Learning.")
print("Cantidad de datos:", len(X))


# ==========================================
# SEPARAR DATOS DE ENTRENAMIENTO Y PRUEBA
# ==========================================

tamano_entrenamiento = int(len(X) * 0.8)

X_train = X.iloc[:tamano_entrenamiento]
X_test = X.iloc[tamano_entrenamiento:]

y_train = y.iloc[:tamano_entrenamiento]
y_test = y.iloc[tamano_entrenamiento:]

print("\nDivisión de los datos:")
print("Datos de entrenamiento:", len(X_train))
print("Datos de prueba:", len(X_test))


# ==========================================
# ENTRENAR MODELO
# ==========================================

escalador = StandardScaler()

X_train_escalado = escalador.fit_transform(X_train)
X_test_escalado = escalador.transform(X_test)

modelo = LogisticRegression(max_iter=1000)

modelo.fit(X_train_escalado, y_train)

print("\nModelo entrenado correctamente.")


# ==========================================
# REALIZAR PREDICCIONES
# ==========================================

predicciones = modelo.predict(X_test_escalado)

print("\nPrimeras 10 predicciones:")
print(predicciones[:10])


# ==========================================
# EVALUAR MODELO
# ==========================================

precision = accuracy_score(y_test, predicciones)

print("\n==========================================")
print("RESULTADOS DEL MODELO")
print("==========================================")

print(f"Precisión del modelo: {precision:.2%}")

print("\nReporte de clasificación:")
print(classification_report(y_test, predicciones))

# ==========================================
# PREPARAR LOS DATOS
# ==========================================

from sklearn.preprocessing import StandardScaler

escalador = StandardScaler()

X_train_escalado = escalador.fit_transform(X_train)

X_test_escalado = escalador.transform(X_test)

print("\nDatos preparados correctamente.")

# ==========================================
# CREAR Y ENTRENAR EL MODELO
# ==========================================

from sklearn.linear_model import LogisticRegression

# Crear el modelo
modelo = LogisticRegression(max_iter=1000)

# Entrenar el modelo con los datos de entrenamiento
modelo.fit(X_train_escalado, y_train)

print("\nModelo entrenado correctamente.")

# ==========================================
# REALIZAR PREDICCIONES
# ==========================================

predicciones = modelo.predict(X_test_escalado)

print("\nPredicciones realizadas correctamente.")
print("Primeras 10 predicciones:")
print(predicciones[:10])

# ==========================================
# EVALUAR EL MODELO
# ==========================================

from sklearn.metrics import accuracy_score, classification_report

# Calcular la precisión
precision = accuracy_score(y_test, predicciones)

print("\n==========================================")
print("RESULTADOS DEL MODELO")
print("==========================================")

print(f"Precisión del modelo: {precision:.2%}")

print("\nReporte de clasificación:")
print(classification_report(y_test, predicciones))



