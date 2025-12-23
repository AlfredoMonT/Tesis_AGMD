import streamlit as st
import pandas as pd
import joblib
import os
import time

# Configuracion de la pagina
st.set_page_config(
    page_title="Sistema de Alerta Temprana - JAQ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
ARCHIVO_MODELO = 'modelo_entrenado_tesis.pkl'
ARCHIVO_DATOS = 'padron_estudiantes_150.csv'
IMAGEN_LOGO = 'logoJAQ.jpg'

# --- FUNCIONES DE CARGA ---
@st.cache_resource
def cargar_modelo():
    if not os.path.exists(ARCHIVO_MODELO):
        return None
    return joblib.load(ARCHIVO_MODELO)

@st.cache_data
def cargar_datos():
    if not os.path.exists(ARCHIVO_DATOS):
        return None
    return pd.read_csv(ARCHIVO_DATOS)

# --- MOTOR DE INTELIGENCIA DEL CHAT ---
def procesar_pregunta_ia(pregunta, df):
    """
    Analiza el lenguaje natural y consulta el DataFrame con lógica ampliada.
    """
    pregunta = pregunta.lower()
    
    # Subconjunto de riesgo para consultas rapidas
    df_riesgo = df[df['Diagnostico_IA'] == 1].copy()
    
    # 1. MAS RIESGOSO
    if "mas riesgoso" in pregunta or "mayor riesgo" in pregunta or "peor caso" in pregunta:
        if df_riesgo.empty: return "No se han detectado estudiantes en riesgo critico actualmente."
        top = df_riesgo.sort_values(by='Probabilidad_Riesgo', ascending=False).iloc[0]
        return f"🚨 **Atencion Prioritaria:** El estudiante con mayor nivel de riesgo es **{top['Nombre_Completo']}** (DNI: {top['DNI']}), con una probabilidad calculada del **{top['Probabilidad_Riesgo']:.1%}**."

    # 2. MENOS RIESGOSO (El mas "sano" o estable)
    elif "menos riesgoso" in pregunta or "menor riesgo" in pregunta or "mas estable" in pregunta:
        # Buscamos en todo el DF, ordenando por riesgo ascendente
        top_sano = df.sort_values(by='Probabilidad_Riesgo', ascending=True).iloc[0]
        return f"✅ El estudiante con indicadores mas estables es **{top_sano['Nombre_Completo']}**, con una probabilidad de riesgo minima del **{top_sano['Probabilidad_Riesgo']:.1%}**."

    # 3. LISTA DE RIESGO
    elif "cuales" in pregunta or "lista" in pregunta or "quienes" in pregunta:
        if df_riesgo.empty: return "No hay estudiantes en riesgo para listar."
        top_5 = df_riesgo.sort_values(by='Probabilidad_Riesgo', ascending=False).head(5)
        respuesta = "📋 **Top 5 Estudiantes que requieren intervencion:**\n\n"
        for i, row in top_5.iterrows():
            respuesta += f"1. **{row['Nombre_Completo']}** ({row['Probabilidad_Riesgo']:.1%})\n"
        return respuesta

    # 4. CANTIDAD
    elif "cuantos" in pregunta or "cantidad" in pregunta or "total" in pregunta:
        total = len(df_riesgo)
        return f"📊 Del total de estudiantes evaluados, he identificado a **{total}** que presentan indicadores de riesgo significativo."

    # 5. INCIDENCIAS (Logica Compuesta)
    elif "incidencias" in pregunta or "factores" in pregunta or "mas problemas" in pregunta:
        # Calculamos un 'Score de Incidencias' al vuelo
        # Sumamos: Pensamientos Suicidas + Historial Familiar + Presion Alta (>4) + Mala Dieta
        df_calc = df.copy()
        df_calc['Incidencias'] = 0
        df_calc['Incidencias'] += df_calc['Have you ever had suicidal thoughts ?'].apply(lambda x: 1 if x == 'Yes' else 0)
        df_calc['Incidencias'] += df_calc['Family History of Mental Illness'].apply(lambda x: 1 if x == 'Yes' else 0)
        df_calc['Incidencias'] += df_calc['Academic Pressure'].apply(lambda x: 1 if x >= 4 else 0)
        df_calc['Incidencias'] += df_calc['Dietary Habits'].apply(lambda x: 1 if x == 'Unhealthy' else 0)
        
        top_inc = df_calc.sort_values(by='Incidencias', ascending=False).iloc[0]
        return (f"⚠️ El estudiante con mas factores de incidencia acumulados es **{top_inc['Nombre_Completo']}**.\n"
                f"Presenta **{top_inc['Incidencias']} factores criticos** simultaneos (como presion alta, antecedentes o habitos nocivos).")

    # 6. Saludo / Default
    else:
        return ("Soy tu asistente virtual del colegio Jose Abelardo Quiñones. Estoy analizando el padron en tiempo real. "
                "Puedes preguntarme:\n"
                "- ¿Quien es el mas riesgoso?\n"
                "- ¿Quien es el menos riesgoso?\n"
                "- ¿Quienes tienen mas incidencias?\n"
                "- ¿Cuantos alumnos estan en alerta?")

# --- PREPROCESAMIENTO ---
def preprocesar_datos(df_entrada):
    df_procesado = df_entrada.copy()
    
    mapa_sueno = {'Less than 5 hours': 4.0, '5-6 hours': 5.5, '7-8 hours': 7.5, 'More than 8 hours': 9.0, 'Others': 6.0}
    mapa_dieta = {'Healthy': 3, 'Moderate': 2, 'Unhealthy': 1, 'Others': 2}
    
    df_procesado['Sleep Duration'] = df_procesado['Sleep Duration'].map(mapa_sueno)
    df_procesado['Dietary Habits'] = df_procesado['Dietary Habits'].map(mapa_dieta)
    
    col_suicida = 'Have you ever had suicidal thoughts ?'
    col_historia = 'Family History of Mental Illness'
    
    df_procesado[col_suicida] = df_procesado[col_suicida].apply(lambda x: 1 if x == 'Yes' else 0)
    df_procesado[col_historia] = df_procesado[col_historia].apply(lambda x: 1 if x == 'Yes' else 0)
    df_procesado['Gender'] = df_procesado['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
    df_procesado = df_procesado.fillna(0)
    
    return df_procesado

# --- INTERFAZ PRINCIPAL ---
def main():
    # BARRA LATERAL (LOGO Y CONTROLES)
    with st.sidebar:
        if os.path.exists(IMAGEN_LOGO):
            st.image(IMAGEN_LOGO, width=200) # Ajusta el ancho segun necesites
        else:
            st.warning("Logo no encontrado (logoJAQ.jpg)")
            
        st.title("Panel de Control")
        st.write("---")
        filtro_riesgo = st.checkbox("🚩 Ver solo Riesgo Alto", value=True)
        st.write("---")
        st.info("Sistema conectado al modelo predictivo v1.0")

    st.title("Sistema de Alerta Temprana - I.E. Jose Abelardo Quiñones")
    
    # 1. Carga
    modelo = cargar_modelo()
    df_original = cargar_datos()
    
    if modelo is None or df_original is None:
        st.error("Error: Faltan archivos del sistema.")
        return

    # 2. Inferencia (Silenciosa)
    df_modelo = preprocesar_datos(df_original)
    cols_modelo = ['Gender', 'Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction',
                   'Sleep Duration', 'Dietary Habits', 'Have you ever had suicidal thoughts ?',
                   'Work/Study Hours', 'Financial Stress', 'Family History of Mental Illness']
    
    df_original['Diagnostico_IA'] = modelo.predict(df_modelo[cols_modelo])
    df_original['Probabilidad_Riesgo'] = modelo.predict_proba(df_modelo[cols_modelo])[:, 1]

    # --- DISEÑO DE COLUMNAS (DASHBOARD | CHAT) ---
    col_dash, col_chat = st.columns([2, 1]) # Proporcion 2:1 (Dashboard mas ancho)

    # === COLUMNA IZQUIERDA: DASHBOARD ===
    with col_dash:
        # Metricas
        riesgo_total = df_original[df_original['Diagnostico_IA'] == 1].shape[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Alumnos", len(df_original))
        m2.metric("En Riesgo", riesgo_total, delta_color="inverse")
        m3.metric("Prevalencia", f"{(riesgo_total/len(df_original))*100:.1f}%")
        
        st.subheader("Padrón de Estudiantes")
        
        # Filtrado para tabla
        df_vista = df_original.copy()
        if filtro_riesgo:
            df_vista = df_vista[df_vista['Diagnostico_IA'] == 1]
            
        # Tabla Simplificada
        df_tabla = df_vista[['DNI', 'Nombre_Completo', 'Promedio_Notas', 'Probabilidad_Riesgo']].copy()
        df_tabla['Probabilidad_Riesgo'] = df_tabla['Probabilidad_Riesgo'].apply(lambda x: f"{x:.1%}")
        df_tabla = df_tabla.rename(columns={'Promedio_Notas': 'Promedio', 'Probabilidad_Riesgo': 'Riesgo IA'})
        
        st.dataframe(df_tabla, use_container_width=True, height=500, hide_index=True)

    # === COLUMNA DERECHA: CHATBOT IA ===
    with col_chat:
        st.markdown("### 🤖 Asistente Virtual")
        st.caption("Consulta directa sobre el padron escolar.")
        
        # Contenedor para el historial de chat (estilo cajita)
        chat_container = st.container(height=500, border=True)
        
        if "mensajes" not in st.session_state:
            st.session_state.mensajes = [{"rol": "assistant", "contenido": "Hola, soy tu asistente virtual del colegio Jose Abelardo Quiñones. ¿En que puedo ayudarte?"}]

        # Input SIEMPRE visible abajo
        pregunta = st.chat_input("Pregunta algo (ej: ¿Quien es el mas riesgoso?)")
        
        # Logica de Chat
        if pregunta:
            st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
            
            # Procesar respuesta
            with st.spinner("Analizando..."):
                time.sleep(0.4)
                respuesta = procesar_pregunta_ia(pregunta, df_original)
                st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})

        # Renderizar historial DENTRO del contenedor
        with chat_container:
            for mensaje in st.session_state.mensajes:
                with st.chat_message(mensaje["rol"]):
                    st.markdown(mensaje["contenido"])

if __name__ == '__main__':
    main()