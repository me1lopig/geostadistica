
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Dashboard Geotecnico Profesional", layout="wide")
st.title("Dashboard geotecnico profesional")
st.caption("Tablas y graficos de parametros en funcion de la profundidad, separados por descripcion de muestra")

PROFILE_KEYS = [
    "Unidad geotécnica", "Descripción Muestra", "Ensayo geotecnia", "Profundidad inicial",
    "Descripción Muestra.1", "Ensayo geotecnia.1", "Profundidad inicial.1",
    "Descripción Muestra.2", "Ensayo geotecnia.2", "Profundidad inicial.2"
]

@st.cache_data
def load_excel(file_obj):
    raw_df = pd.read_excel(file_obj, sheet_name=0)
    raw_df.columns = [str(col).strip() for col in raw_df.columns]
    return raw_df

@st.cache_data
def clean_value(series_vals):
    as_text = series_vals.astype(str).str.strip()
    as_text = as_text.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    as_text = as_text.str.replace("%", "", regex=False)
    as_text = as_text.str.replace(" ", "", regex=False)
    return pd.to_numeric(as_text.str.replace(",", ".", regex=False), errors="coerce")

@st.cache_data
def build_long_database(raw_df):
    df_local = raw_df.copy()
    for col_name in df_local.columns:
        if df_local[col_name].dtype == object:
            df_local[col_name] = df_local[col_name].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})

    group_specs = [
        {"desc": "Descripción Muestra", "ensayo": "Ensayo geotecnia", "prof": "Profundidad inicial", "suffix": ""},
        {"desc": "Descripción Muestra.1", "ensayo": "Ensayo geotecnia.1", "prof": "Profundidad inicial.1", "suffix": ".1"},
        {"desc": "Descripción Muestra.2", "ensayo": "Ensayo geotecnia.2", "prof": "Profundidad inicial.2", "suffix": ".2"}
    ]

    long_frames = []
    for group_item in group_specs:
        if group_item["desc"] not in df_local.columns or group_item["prof"] not in df_local.columns:
            continue
        base_cols = []
        for col_name in df_local.columns:
            if col_name in PROFILE_KEYS:
                continue
            if group_item["suffix"] == "":
                if not col_name.endswith(".1") and not col_name.endswith(".2"):
                    base_cols.append(col_name)
            else:
                if col_name.endswith(group_item["suffix"]):
                    base_cols.append(col_name)
        temp_df = pd.DataFrame()
        temp_df["descripcion_muestra"] = df_local[group_item["desc"]] if group_item["desc"] in df_local.columns else np.nan
        temp_df["ensayo"] = df_local[group_item["ensayo"]] if group_item["ensayo"] in df_local.columns else np.nan
        temp_df["profundidad"] = clean_value(df_local[group_item["prof"]])
        temp_df["unidad_geotecnica"] = df_local["Unidad geotécnica"] if "Unidad geotécnica" in df_local.columns else np.nan
        temp_df["row_id"] = np.arange(len(df_local))
        temp_df = pd.concat([temp_df, df_local[base_cols]], axis=1)
        rename_map = {}
        for col_name in base_cols:
            if group_item["suffix"] != "" and col_name.endswith(group_item["suffix"]):
                rename_map[col_name] = col_name.replace(group_item["suffix"], "")
        temp_df = temp_df.rename(columns=rename_map)
        value_cols = [c for c in temp_df.columns if c not in ["descripcion_muestra", "ensayo", "profundidad", "unidad_geotecnica", "row_id"]]
        melted_df = temp_df.melt(id_vars=["descripcion_muestra", "ensayo", "profundidad", "unidad_geotecnica", "row_id"], value_vars=value_cols, var_name="parametro", value_name="valor_raw")
        melted_df["valor"] = clean_value(melted_df["valor_raw"])
        melted_df = melted_df.dropna(subset=["descripcion_muestra", "parametro", "valor"], how="any")
        long_frames.append(melted_df[["descripcion_muestra", "ensayo", "profundidad", "unidad_geotecnica", "row_id", "parametro", "valor"]])

    if long_frames:
        long_df = pd.concat(long_frames, ignore_index=True)
    else:
        long_df = pd.DataFrame(columns=["descripcion_muestra", "ensayo", "profundidad", "unidad_geotecnica", "row_id", "parametro", "valor"])

    long_df = long_df.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    long_df = long_df.dropna(subset=["descripcion_muestra", "profundidad", "valor"], how="any")
    return long_df

uploaded = st.file_uploader("Sube el archivo Excel", type=["xlsx", "xls"])
if uploaded is None:
    st.info("Sube un Excel para iniciar el dashboard")
    st.stop()

raw_df = load_excel(uploaded)
long_df = build_long_database(raw_df)

if long_df.empty:
    st.error("No se pudieron generar datos numericos normalizados a partir del archivo")
    st.dataframe(raw_df.head(20), use_container_width=True)
    st.stop()

with st.sidebar:
    st.header("Filtros")
    desc_list = sorted(long_df["descripcion_muestra"].dropna().unique().tolist())
    ensayo_list = sorted(long_df["ensayo"].dropna().unique().tolist())
    parametro_list = sorted(long_df["parametro"].dropna().unique().tolist())
    selected_desc = st.multiselect("Descripcion muestra", desc_list, default=desc_list[:min(8, len(desc_list))])
    selected_ensayo = st.multiselect("Ensayo", ensayo_list, default=ensayo_list)
    selected_param = st.multiselect("Parametros", parametro_list, default=parametro_list[:min(8, len(parametro_list))])
    prof_min = float(long_df["profundidad"].min())
    prof_max = float(long_df["profundidad"].max())
    selected_prof = st.slider("Rango profundidad", min_value=prof_min, max_value=prof_max, value=(prof_min, prof_max))

filtered_long = long_df.copy()
if selected_desc:
    filtered_long = filtered_long[filtered_long["descripcion_muestra"].isin(selected_desc)]
if selected_ensayo:
    filtered_long = filtered_long[filtered_long["ensayo"].isin(selected_ensayo)]
if selected_param:
    filtered_long = filtered_long[filtered_long["parametro"].isin(selected_param)]
filtered_long = filtered_long[(filtered_long["profundidad"] >= selected_prof[0]) & (filtered_long["profundidad"] <= selected_prof[1])]

summary_a, summary_b, summary_c, summary_d = st.columns(4)
summary_a.metric("Registros normalizados", int(filtered_long.shape[0]))
summary_b.metric("Descripciones muestra", int(filtered_long["descripcion_muestra"].nunique()))
summary_c.metric("Parametros", int(filtered_long["parametro"].nunique()))
summary_d.metric("Ensayos", int(filtered_long["ensayo"].nunique()))

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Resumen", "Tabla", "Graficos", "Perfiles", "Exportacion"])

with tab1:
    st.subheader("Vista ejecutiva")
    top_params = filtered_long["parametro"].value_counts().reset_index()
    top_params.columns = ["parametro", "conteo"]
    st.dataframe(top_params, use_container_width=True)
    if not top_params.empty:
        fig_top = px.bar(top_params.head(15), x="conteo", y="parametro", orientation="h", title="Parametros con mas registros")
        fig_top.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_top, use_container_width=True)
    depth_desc = filtered_long.groupby("descripcion_muestra")["profundidad"].agg(["min", "max", "count"]).reset_index()
    st.dataframe(depth_desc, use_container_width=True)

with tab2:
    st.subheader("Tabla detallada")
    pivot_df = filtered_long.pivot_table(index=["descripcion_muestra", "ensayo", "profundidad", "row_id"], columns="parametro", values="valor", aggfunc="first").reset_index()
    st.dataframe(pivot_df, use_container_width=True, height=500)

with tab3:
    st.subheader("Graficos interactivos")
    for param_name in selected_param[:12]:
        chart_df = filtered_long[filtered_long["parametro"] == param_name].copy()
        if chart_df.empty:
            continue
        chart_df = chart_df.sort_values("profundidad")
        fig_scatter = px.scatter(chart_df, x="valor", y="profundidad", color="descripcion_muestra", symbol="ensayo", title=param_name + " vs profundidad", hover_data=["descripcion_muestra", "ensayo", "unidad_geotecnica"])
        fig_scatter.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_scatter, use_container_width=True)
        stat_df = chart_df.groupby("descripcion_muestra")["valor"].agg(["count", "mean", "median", "min", "max", "std"]).reset_index()
        st.dataframe(stat_df, use_container_width=True)
    if selected_param:
        box_df = filtered_long[filtered_long["parametro"].isin(selected_param[:6])]
        if not box_df.empty:
            fig_box = px.box(box_df, x="parametro", y="valor", color="descripcion_muestra", title="Distribucion por parametro")
            st.plotly_chart(fig_box, use_container_width=True)

with tab4:
    st.subheader("Perfiles por descripcion de muestra")
    for desc_name in selected_desc[:12]:
        desc_df = filtered_long[filtered_long["descripcion_muestra"] == desc_name]
        if desc_df.empty:
            continue
        st.markdown("#### " + str(desc_name))
        profile_table = desc_df.pivot_table(index=["profundidad", "ensayo"], columns="parametro", values="valor", aggfunc="first").reset_index().sort_values("profundidad")
        st.dataframe(profile_table, use_container_width=True, height=240)
        param_subset = [p for p in selected_param[:4] if p in desc_df["parametro"].unique().tolist()]
        if param_subset:
            fig_profile = go.Figure()
            for param_name in param_subset:
                line_df = desc_df[desc_df["parametro"] == param_name].sort_values("profundidad")
                fig_profile.add_trace(go.Scatter(x=line_df["valor"], y=line_df["profundidad"], mode="lines+markers", name=param_name))
            fig_profile.update_layout(title="Perfil combinado - " + str(desc_name), xaxis_title="Valor", yaxis_title="Profundidad")
            fig_profile.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_profile, use_container_width=True)

with tab5:
    st.subheader("Exportacion")
    export_wide = filtered_long.pivot_table(index=["descripcion_muestra", "ensayo", "profundidad", "row_id"], columns="parametro", values="valor", aggfunc="first").reset_index()
    csv_long = filtered_long.to_csv(index=False).encode("utf-8")
    csv_wide = export_wide.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar base normalizada", data=csv_long, file_name="base_geotecnia_normalizada.csv", mime="text/csv")
    st.download_button("Descargar tabla pivotada", data=csv_wide, file_name="tabla_geotecnia_pivotada.csv", mime="text/csv")
    st.markdown("La base normalizada deja cada parametro en filas. La pivotada deja los parametros en columnas para usar en Excel o Power BI.")
