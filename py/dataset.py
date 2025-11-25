import pandas as pd
import numpy as np
import random

# Semilla para reproducibilidad
np.random.seed(42)

# Cantidad de alumnos simulados
n_alumnos = 500

data = {
    'Edad': np.random.randint(13, 18, n_alumnos),
    'Sexo': np.random.choice([0, 1], n_alumnos), # 0=M, 1=F
    # Notas más realistas: Distibución normal centrada en 14
    'Promedio_General': np.clip(np.random.normal(14, 2.5, n_alumnos), 0, 20).round(1),
    'Num_Cursos_Desaprobados': np.random.choice([0, 1, 2, 3, 4], n_alumnos, p=[0.6, 0.2, 0.1, 0.05, 0.05]),
    'Porcentaje_Asistencia': np.clip(np.random.normal(90, 10, n_alumnos), 0, 100).round(1),
    'Num_Tardanzas': np.random.poisson(3, n_alumnos), # Distribución de Poisson para conteos
    'Antecedentes_Psic': np.random.choice([0, 1], n_alumnos, p=[0.85, 0.15]),
    # Simulamos el puntaje GAD-7 (0-21)
    'Resultado_Cuestionario_GAD7': np.random.randint(0, 22, n_alumnos)
}

df = pd.DataFrame(data)

# Lógica de negocio para asignar la etiqueta "Ansiedad_Riesgo" (Variable Dependiente)
# Esto simula que un psicólogo los etiquetó basándose en reglas lógicas
def asignar_riesgo(row):
    score = 0
    if row['Resultado_Cuestionario_GAD7'] >= 10: score += 5
    if row['Antecedentes_Psic'] == 1: score += 3
    if row['Promedio_General'] < 11: score += 2
    if row['Num_Tardanzas'] > 8: score += 2
    
    # Umbral arbitrario para la simulación
    return 1 if score >= 5 else 0

df['Ansiedad_Riesgo'] = df.apply(asignar_riesgo, axis=1)

# Guardar
df.to_csv('dataset_tesis_simulado.csv', index=False)
print("Dataset generado exitosamente.")