import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# =============================================================================
# 1. DEFINICIÓN DEL DATASET (Simulación basada en la Tabla de Variables, Pág. 19)
# =============================================================================
# Se crea un DataFrame con datos sintéticos para demostrar la funcionalidad.
# En la implementación real, estos datos vendrían del CSV recopilado en la I.E.

data_simulada = {
    'Edad': [14, 16, 15, 17, 15, 14],
    'Sexo': [1, 0, 1, 1, 0, 0],  # 1=Masculino, 0=Femenino
    'Promedio_General': [16.5, 11.0, 10.5, 14.0, 18.0, 12.0],
    'Num_Cursos_Desaprobados': [0, 3, 4, 1, 0, 2],
    'Porcentaje_Asistencia': [98.0, 70.5, 65.0, 90.0, 99.0, 85.0],
    'Num_Tardanzas': [2, 15, 20, 5, 0, 8],
    'Antecedentes_Psic': [0, 1, 0, 0, 0, 1],  # 1=Sí, 0=No
    'Resultado_Cuestionario_GAD7': [3, 18, 15, 8, 2, 12], # Score del test
    # Variable Objetivo (Target): 0=Bajo Riesgo, 1=Alto Riesgo
    'Ansiedad_Riesgo': [0, 1, 1, 0, 0, 1]
}

df_entrenamiento = pd.DataFrame(data_simulada)

# =============================================================================
# 2. ENTRENAMIENTO DEL MODELO (Prototipo)
# =============================================================================
# Selección de variables predictoras (X) y variable objetivo (y)
X = df_entrenamiento.drop('Ansiedad_Riesgo', axis=1)
y = df_entrenamiento['Ansiedad_Riesgo']

# Instanciamos el modelo (Random Forest, sugerido en el marco teórico)
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X, y)

print(">> Modelo entrenado exitosamente con el dataset piloto.\n")

# =============================================================================
# 3. SIMULACIÓN DE UN CASO REAL (Inferencia)
# =============================================================================
# Datos de un nuevo estudiante (Ejemplo: "Alumno X" de 4to de Secundaria)
# Caso: Alumno con notas regulares, algunas tardanzas y puntaje GAD-7 moderado.

nuevo_estudiante = pd.DataFrame([{
    'Edad': 15,
    'Sexo': 1,                       # Masculino
    'Promedio_General': 12.8,        # Promedio regular
    'Num_Cursos_Desaprobados': 1,    # 1 curso jalado
    'Porcentaje_Asistencia': 88.5,   # Asistencia aceptable
    'Num_Tardanzas': 6,              # Algunas tardanzas
    'Antecedentes_Psic': 0,          # Sin antecedentes previos
    'Resultado_Cuestionario_GAD7': 14 # Puntaje límite (Ansiedad Moderada)
}])

# Predicción de clase (0 o 1) y probabilidad (%)
prediccion = modelo.predict(nuevo_estudiante)
probabilidad = modelo.predict_proba(nuevo_estudiante)

# =============================================================================
# 4. SALIDA DEL SISTEMA
# =============================================================================
resultado_texto = "ALTO RIESGO" if prediccion[0] == 1 else "BAJO RIESGO"
confianza = probabilidad[0][1] * 100  # Probabilidad de ser clase 1

print("--- REPORTE DE DETECCIÓN TEMPRANA ---")
print(f"Datos del Estudiante: Promedio {nuevo_estudiante['Promedio_General'][0]}, "
      f"GAD-7 Score: {nuevo_estudiante['Resultado_Cuestionario_GAD7'][0]}")
print(f"Diagnóstico del Modelo: {resultado_texto}")
print(f"Probabilidad Calculada de Ansiedad: {confianza:.2f}%")

# Regla de negocio para el psicólogo
if prediccion[0] == 1:
    print(">> ACCIÓN RECOMENDADA: Programar cita con psicología escolar.")
else:
    print(">> ACCIÓN RECOMENDADA: Monitoreo regular.")