import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# ==========================================
# CONFIGURACIÓN
# ==========================================
ARCHIVO_MODELO = 'modelo_entrenado_tesis.pkl'
ARCHIVO_DATASET = 'Student Depression Dataset.csv'

def main():
    print("--- INICIANDO AUDITORÍA DEL MODELO ---")
    
    # 1. VERIFICAR ARCHIVOS
    if not os.path.exists(ARCHIVO_MODELO):
        print(f"ERROR: No se encuentra el modelo '{ARCHIVO_MODELO}'. Ejecuta primero el entrenamiento.")
        return
    
    if not os.path.exists(ARCHIVO_DATASET):
        print(f"ERROR: No se encuentra el dataset '{ARCHIVO_DATASET}'.")
        return

    # 2. CARGAR Y PREPROCESAR DATOS (Idéntico al entrenamiento para ser justo)
    print("1. Cargando y procesando datos originales...")
    df = pd.read_csv(ARCHIVO_DATASET)
    
    # Filtro de estudiantes
    df = df[df['Profession'] == 'Student'].copy()
    
    # Mapeos
    sleep_map = {'Less than 5 hours': 4.0, '5-6 hours': 5.5, '7-8 hours': 7.5, 'More than 8 hours': 9.0, 'Others': 6.0}
    diet_map = {'Healthy': 3, 'Moderate': 2, 'Unhealthy': 1, 'Others': 2}
    
    df['Sleep Duration'] = df['Sleep Duration'].map(sleep_map)
    df['Dietary Habits'] = df['Dietary Habits'].map(diet_map)
    
    # Binarios
    cols_bin = ['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
    for col in cols_bin:
        df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)
        
    df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
    df = df.fillna(df.median(numeric_only=True))
    
    # Definir X e y
    features = ['Gender', 'Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 
                'Sleep Duration', 'Dietary Habits', 'Have you ever had suicidal thoughts ?', 
                'Work/Study Hours', 'Financial Stress', 'Family History of Mental Illness']
    target = 'Depression'
    
    X = df[features]
    y = df[target]
    
    # 3. RECUPERAR EL CONJUNTO DE PRUEBA (TEST SET)
    # Usamos la misma semilla (42) para asegurar que evaluamos en datos que el modelo NO vio
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"2. Evaluando modelo sobre {len(X_test)} estudiantes desconocidos...")
    
    # 4. CARGAR MODELO Y PREDECIR
    modelo = joblib.load(ARCHIVO_MODELO)
    y_pred = modelo.predict(X_test)
    
    # 5. CÁLCULO DE MÉTRICAS CLÍNICAS
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Fórmulas
    accuracy = accuracy_score(y_test, y_pred)
    sensibilidad = tp / (tp + fn) # Recall (Capacidad de detectar enfermos)
    especificidad = tn / (tn + fp) # Capacidad de descartar sanos
    vpp = tp / (tp + fp) if (tp + fp) > 0 else 0 # Valor Predictivo Positivo
    vpn = tn / (tn + fn) if (tn + fn) > 0 else 0 # Valor Predictivo Negativo
    
    # 6. REPORTE FINAL
    print("\n" + "="*50)
    print("RESULTADOS DE LA EVALUACIÓN (Tesis Cap. 4)")
    print("="*50)
    
    print("\n--- MATRIZ DE CONFUSIÓN ---")
    print(f"                  Predicho: SANO (0)   Predicho: RIESGO (1)")
    print(f"Real: SANO (0)    {tn:<20} {fp} (Falsos Positivos)")
    print(f"Real: RIESGO (1)  {fn:<20} {tp} (Aciertos de Riesgo)")
    
    print("\n--- MÉTRICAS TÉCNICAS ---")
    print(f"1. EXACTITUD (Accuracy):      {accuracy:.2%}  (Rendimiento global)")
    print(f"2. SENSIBILIDAD (Recall):     {sensibilidad:.2%}  <-- IMPORTANTE: Capacidad de detección")
    print(f"3. ESPECIFICIDAD:             {especificidad:.2%}  (Capacidad de descarte)")
    
    print("\n--- INTERPRETACIÓN PARA LA TESIS ---")
    print(f"> El sistema detecta correctamente a {int(sensibilidad*100)} de cada 100 estudiantes con ansiedad.")
    print(f"> El sistema evita falsas alarmas en {int(especificidad*100)} de cada 100 estudiantes sanos.")
    
    # Guardar en archivo de texto para copiar y pegar
    with open("resultados_auditoria.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\nSensibilidad: {sensibilidad:.4f}\nEspecificidad: {especificidad:.4f}")
    print("\n(Se ha generado el archivo 'resultados_auditoria.txt' con estos datos)")

if __name__ == "__main__":
    main()