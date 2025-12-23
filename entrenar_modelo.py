import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# 1. CARGA DE DATOS
print("Cargando dataset...")
df = pd.read_csv('Student Depression Dataset.csv')

# FILTRO CLAVE: Usamos solo 'Student' para evitar sesgos de otras profesiones
df = df[df['Profession'] == 'Student'].copy()

# 2. PREPROCESAMIENTO (Limpieza para que la IA entienda)
# Mapeo de hábitos de sueño a valores numéricos
sleep_map = {
    'Less than 5 hours': 4.0,
    '5-6 hours': 5.5,
    '7-8 hours': 7.5,
    'More than 8 hours': 9.0,
    'Others': 6.0
}
df['Sleep Duration'] = df['Sleep Duration'].map(sleep_map)

# Mapeo de hábitos alimenticios
diet_map = {'Healthy': 3, 'Moderate': 2, 'Unhealthy': 1, 'Others': 2}
df['Dietary Habits'] = df['Dietary Habits'].map(diet_map)

# Convertir Si/No a 1/0
binary_cols = ['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
for col in binary_cols:
    df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)

# Codificar Género (Male=1, Female=0 - o viceversa, lo hace automático)
df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

# Seleccionamos las columnas que usará el colegio
features = ['Gender', 'Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 
            'Sleep Duration', 'Dietary Habits', 'Have you ever had suicidal thoughts ?', 
            'Work/Study Hours', 'Financial Stress', 'Family History of Mental Illness']
target = 'Depression'

# Llenar vacíos (si los hay) con la mediana
df = df.fillna(df.median(numeric_only=True))

X = df[features]
y = df[target]

# 3. ENTRENAMIENTO
print("Entrenando modelo Random Forest...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# 4. EVALUACIÓN
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n--- RESULTADOS ---")
print(f"Precisión Global (Accuracy): {acc:.2%}")
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# 5. GUARDADO DEL MODELO
print("\nGuardando el 'cerebro' del modelo...")
joblib.dump(rf_model, 'modelo_entrenado_tesis.pkl') 
print("¡Listo! Archivo 'modelo_entrenado_tesis.pkl' generado.")