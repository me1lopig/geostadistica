import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(page_title="Estratígrafo IA", layout="wide")

st.title("🤖 Estratígrafo Inteligente (K-Means & Jerárquico)")
st.markdown("""
Esta aplicación identifica unidades geotécnicas automáticamente. 
Puedes usar **K-Means** para un análisis rápido por defecto, o cambiar a **Clustering Jerárquico** para buscar estratos con mejor continuidad vertical.
""")

# 1. CARGA DE DATOS
uploaded_file = st.file_uploader("Sube tu archivo Excel o CSV (ListadoLab)", type=["csv", "xlsx"])

if uploaded_file:
    # Leer el archivo (Manejo de decimales y separadores)
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, decimal=",")
    else:
        df = pd.read_excel(uploaded_file)

    st.success("Archivo cargado correctamente.")

    # 2. PREPROCESAMIENTO Y LIMPIEZA
    cols_interes = {
        'Descripción Muestra': 'Sondeo',
        'Profundidad inicial': 'Profundidad',
        'SPT (valores centrales)': 'N_SPT',
        'Tamiz Finos': 'Finos',
        'LL': 'LL',
        'IP': 'IP',
        'Humedad': 'Humedad'
    }

    data = df[list(cols_interes.keys())].copy()
    data = data.rename(columns=cols_interes)

    for col in ['Profundidad', 'N_SPT', 'Finos', 'LL', 'IP', 'Humedad']:
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', '.'), errors='coerce')

    # 3. INTERPOLACIÓN DE NULOS
    data = data.sort_values(by=['Sondeo', 'Profundidad'])
    data_clean = data.groupby('Sondeo').apply(lambda x: x.interpolate(method='linear').ffill().bfill())
    
    # Eliminamos el nivel extra del groupby y las filas 100% nulas
    data_clean = data_clean.reset_index(drop=True)
    data_final = data_clean.dropna(subset=['N_SPT', 'Finos', 'Humedad'])

    # 4. CONFIGURACIÓN DEL MODELO EN EL SIDEBAR
    st.sidebar.header("⚙️ Configuración de la IA")
    
    # NUEVO: Selector de Algoritmo
    algoritmo = st.sidebar.radio(
        "Selecciona el Algoritmo de Agrupamiento:",
        ("K-Means (Por defecto)", "Jerárquico (Aglomerativo)")
    )
    
    n_clusters = st.sidebar.slider("Número de Unidades Geotécnicas (Clusters)", 2, 8, 4)
    features = st.sidebar.multiselect(
        "Variables para el agrupamiento:",
        ['N_SPT', 'Finos', 'LL', 'IP', 'Humedad', 'Profundidad'],
        default=['N_SPT', 'Finos', 'Humedad', 'Profundidad']
    )

    if len(features) > 0:
        # 5. MACHINE LEARNING: ESCALADO Y CLUSTERING
        scaler = StandardScaler()
        # Le damos más peso a la profundidad artificialmente para forzar estratos continuos
        X_scaled = scaler.fit_transform(data_final[features])
        
        # Lógica de Selección de Algoritmo
        if algoritmo == "K-Means (Por defecto)":
            model = KMeans(n_clusters=n_clusters, random_state=42)
        else:
            # Clustering Jerárquico minimizando la varianza dentro del clúster (Ward)
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            
        # Entrenar y predecir
        data_final['Unidad_Geotecnica'] = model.fit_predict(X_scaled)
        data_final['Unidad_Geotecnica'] = data_final['Unidad_Geotecnica'].astype(str)

        # 6. VISUALIZACIÓN INTERACTIVA
        st.subheader(f"📊 Perfil Estratigráfico usando {algoritmo.split(' ')[0]}")
        
        fig = px.scatter(
            data_final, 
            x="Sondeo", 
            y="Profundidad", 
            color="Unidad_Geotecnica",
            hover_data=features,
            title=f"Agrupamiento con {algoritmo.split(' ')[0]}",
            symbol="Unidad_Geotecnica",
            height=700
        )
        
        fig.update_yaxes(autorange="reversed") 
        fig.update_layout(legend_title_text='Unidad Geotécnica')
        
        st.plotly_chart(fig, use_container_width=True)

        # 7. ANÁLISIS DE LAS UNIDADES
        st.subheader("📋 Propiedades Medias por Unidad")
        resumen = data_final.groupby('Unidad_Geotecnica')[features].mean()
        st.dataframe(resumen.style.highlight_max(axis=0, color='#ffaa00'))

    else:
        st.warning("Selecciona al menos una variable en el panel de la izquierda.")

else:
    st.info("👋 Sube tu archivo CSV o Excel de laboratorio para empezar.")