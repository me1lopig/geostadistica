import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Visor Geotécnico", page_icon="🪨", layout="wide")

PALETTE = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel1 + px.colors.qualitative.Safe

def is_good_id(val):
    if pd.isna(val):
        return False
    s = str(val).strip()
    # Añadido el filtro para "none"
    if s == "" or s.lower() == "nan" or s.lower() == "none":
        return False
    return True

def clean_num(series):
    s = series.astype(str).str.replace(",", ".", regex=False)
    s = s.str.replace("nan", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

@st.cache_data
def load_data(file):
    df0 = pd.read_excel(file, sheet_name=0)

    # Asegurar nombres de columnas únicos
    new_cols = []
    seen = {}
    for c in df0.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(c + "_" + str(seen[c]))
        else:
            seen[c] = 0
            new_cols.append(str(c)) # Forzar a string por si hay columnas numéricas
    df0.columns = new_cols

    # Convertir columnas que parecen numéricas
    hints = [
        "Profundidad", "ISPT", "SPT", "MI",
        "Tamiz", "20", "5", "2", "0.4", "0.08",
        "LL", "LP", "IP",
        "Densidad", "Peso", "Humedad",
        "RCS", "Rozamiento", "Angulo", "Ángulo", "Cohesi",
        "Indice", "Índice", "Presi", "kPa", "KPa",
        "Sulfat", "acidez", "CO3", "%", "CBR", "KN", "kg"
    ]

    for c in df0.columns:
        if any(h.lower() in c.lower() for h in hints):
            try:
                df0[c] = clean_num(df0[c])
            except Exception:
                pass

    return df0

def detect_columns(df):
    muestra_col = next(
        (c for c in df.columns if "descripci" in c.lower() and "muestra" in c.lower() and "." not in c),
        None
    )
    if muestra_col is None:
        muestra_col = next((c for c in df.columns if "muestra" in c.lower() and "." not in c), df.columns[1] if len(df.columns) > 1 else df.columns[0])

    depth_candidates = [c for c in df.columns if "profundidad" in c.lower() and "inicial" in c.lower() and "." not in c]
    if len(depth_candidates) > 0:
        depth_col = depth_candidates[0]
    else:
        depth_col = next((c for c in df.columns if "profundidad" in c.lower()), df.columns[3] if len(df.columns) > 3 else df.columns[0])

    ensayo_col = next((c for c in df.columns if c.lower().strip() == "ensayo geotecnia"), None)
    uscs_col = next((c for c in df.columns if "uscs" in c.lower()), None)

    return muestra_col, depth_col, ensayo_col, uscs_col

def group_numeric_columns(df, depth_col):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    groups = {
        "SPT y penetración": [],
        "Granulometría": [],
        "Plasticidad": [],
        "Densidad y humedad": [],
        "Resistencia y corte": [],
        "Consolidación": [],
        "Química y otros": [],
        "Otros": []
    }

    for c in num_cols:
        if c == depth_col:
            continue
        cl = c.lower()

        if "ispt" in cl or "spt" in cl or cl == "mi" or " mi" in cl or cl.startswith("mi "):
            groups["SPT y penetración"].append(c)
        elif "tamiz" in cl or c in ["20", "5", "2", "0.4", "0.08"]:
            groups["Granulometría"].append(c)
        elif cl in ["ll", "lp", "ip"] or "atterberg" in cl:
            groups["Plasticidad"].append(c)
        elif "densidad" in cl or "peso espec" in cl or "humedad" in cl:
            groups["Densidad y humedad"].append(c)
        elif "rcs" in cl or "cohes" in cl or "rozamiento" in cl or "angulo" in cl or "ángulo" in cl:
            groups["Resistencia y corte"].append(c)
        elif "preconsolid" in cl or "hinch" in cl or "poros" in cl:
            groups["Consolidación"].append(c)
        elif "sulfat" in cl or "acidez" in cl or "co3" in cl:
            groups["Química y otros"].append(c)
        else:
            groups["Otros"].append(c)

    groups = {k: v for k, v in groups.items() if len(v) > 0}
    return groups

def plot_param_depth(df, depth_col, value_col, title):
    d = df[[depth_col, value_col]].dropna()
    fig = go.Figure()

    if d.empty:
        fig.update_layout(title=title + " (sin datos)", height=420)
        return fig

    d = d.sort_values(depth_col)
    fig.add_trace(
        go.Scatter(
            x=d[value_col],
            y=d[depth_col],
            mode="markers",
            marker=dict(size=10, color="#2e6da4", line=dict(width=1, color="white")),
            hovertemplate=value_col + ": %{x}<br>Profundidad: %{y} m<extra></extra>"
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1a3a5c")),
        xaxis=dict(title=value_col, gridcolor="#e8edf2", zeroline=False),
        yaxis=dict(title="Profundidad (m)", gridcolor="#e8edf2", autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(l=60, r=30, t=55, b=50)
    )
    return fig

def casagrande_plot(df, ll_col, ip_col, depth_col=None, uscs_col=None):
    cols = [ll_col, ip_col]
    if depth_col and depth_col in df.columns:
        cols.append(depth_col)
    if uscs_col and uscs_col in df.columns:
        cols.append(uscs_col)

    sub = df[cols].dropna(subset=[ll_col, ip_col])
    fig = go.Figure()

    if sub.empty:
        fig.update_layout(title="Carta de Plasticidad (sin datos)", height=520)
        return fig

    ll_arr = np.linspace(0, 110, 300)
    fig.add_trace(go.Scatter(x=ll_arr, y=0.73 * (ll_arr - 20), mode="lines", name="Línea A", line=dict(color="#e74c3c", dash="dash", width=2)))
    fig.add_trace(go.Scatter(x=ll_arr, y=0.9 * (ll_arr - 8), mode="lines", name="Línea U", line=dict(color="#f39c12", dash="dot", width=1.5)))

    custom = None
    hover = "LL: %{x:.1f}<br>IP: %{y:.1f}"

    if depth_col and depth_col in sub.columns:
        custom = sub[[depth_col]].values
        hover = hover + "<br>Prof: %{customdata[0]:.2f} m"

    if uscs_col and uscs_col in sub.columns:
        uscs_vals = sub[[uscs_col]].astype(str).values
        if custom is None:
            custom = uscs_vals
            hover = hover + "<br>USCS: %{customdata[0]}"
        else:
            custom = np.concatenate([custom, uscs_vals], axis=1)
            hover = hover + "<br>USCS: %{customdata[1]}"

    fig.add_trace(
        go.Scatter(
            x=sub[ll_col],
            y=sub[ip_col],
            mode="markers",
            name="Datos",
            marker=dict(size=11, color="#2e6da4", line=dict(width=1, color="white")),
            customdata=custom,
            hovertemplate=hover + "<extra></extra>"
        )
    )

    fig.update_layout(
        title=dict(text="Carta de Plasticidad de Casagrande", font=dict(size=14, color="#1a3a5c")),
        xaxis=dict(title="Límite Líquido (%)", range=[0, 110], gridcolor="#e8edf2"),
        yaxis=dict(title="Índice de Plasticidad (%)", range=[0, 80], gridcolor="#e8edf2"),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#ccc", borderwidth=1),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=520,
        margin=dict(l=60, r=30, t=55, b=50)
    )

    return fig

def uscs_pie(df, uscs_col):
    fig = go.Figure()

    if uscs_col is None or uscs_col not in df.columns:
        fig.update_layout(title="USCS (sin columna)", height=420)
        return fig

    s = df[uscs_col].dropna().astype(str)
    s = s[s.str.strip().str.lower() != "nan"]
    vc = s.value_counts()

    if vc.empty:
        fig.update_layout(title="USCS (sin datos)", height=420)
        return fig

    fig.add_trace(
        go.Pie(
            labels=vc.index,
            values=vc.values,
            marker=dict(colors=PALETTE[:len(vc)], line=dict(color="white", width=2)),
            textinfo="label+percent+value",
            hole=0.35
        )
    )

    fig.update_layout(
        title=dict(text="Distribución USCS", font=dict(size=14, color="#1a3a5c")),
        paper_bgcolor="white",
        height=420,
        margin=dict(l=30, r=30, t=55, b=30)
    )

    return fig

def render_group(df, depth_col, cols):
    cols_ok = [c for c in cols if df[c].notna().sum() > 0]
    if len(cols_ok) == 0:
        st.info("No hay parámetros con datos en este grupo para la selección actual.")
        return

    for i in range(0, len(cols_ok), 2):
        c1, c2 = st.columns(2)
        col_a = cols_ok[i]
        with c1:
            st.plotly_chart(plot_param_depth(df, depth_col, col_a, col_a + " vs Profundidad"), use_container_width=True)

        if i + 1 < len(cols_ok):
            col_b = cols_ok[i + 1]
            with c2:
                st.plotly_chart(plot_param_depth(df, depth_col, col_b, col_b + " vs Profundidad"), use_container_width=True)

def numeric_summary_table(df, depth_col):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != depth_col]
    out = []

    for c in num_cols:
        nn = int(df[c].notna().sum())
        if nn == 0:
            continue

        out.append({
            "Parámetro": c,
            "N": nn,
            "Min": float(df[c].min()),
            "P25": float(df[c].quantile(0.25)),
            "Mediana": float(df[c].median()),
            "P75": float(df[c].quantile(0.75)),
            "Max": float(df[c].max())
        })

    if len(out) == 0:
        return pd.DataFrame(columns=["Parámetro", "N", "Min", "P25", "Mediana", "P75", "Max"])

    summ_df = pd.DataFrame(out).sort_values(["N", "Parámetro"], ascending=[False, True])
    return summ_df

# ------------------------------
# UI PRINCIPAL
# ------------------------------
st.title("Visor de Ensayos Geotécnicos")
st.caption("Sube un Excel con la misma estructura de columnas y explora los parámetros por profundidad.")

with st.sidebar:
    st.header("Entrada")
    uploaded = st.file_uploader("Sube un Excel", type=["xlsx", "xls"])
    st.markdown("- La app detecta automáticamente columna de muestra, profundidad, USCS.")
    st.markdown("- Si tu Excel tiene varias hojas, se usa la primera.")

if uploaded is None:
    st.info("Sube tu archivo Excel para comenzar.")
    st.stop()

# Cargar datos y detectar columnas iniciales
df_all = load_data(uploaded)
m_col_auto, d_col_auto, ensayo_col, uscs_col = detect_columns(df_all)

# --- NUEVO: Selectores manuales de columnas ---
with st.sidebar:
    st.markdown("---")
    st.header("⚙️ Ajuste de Columnas")
    st.caption("Verifica o cambia las columnas detectadas:")
    
    idx_m = list(df_all.columns).index(m_col_auto) if m_col_auto in df_all.columns else 0
    idx_d = list(df_all.columns).index(d_col_auto) if d_col_auto in df_all.columns else 0
    
    muestra_col = st.selectbox("Columna de Muestra", df_all.columns, index=idx_m)
    depth_col = st.selectbox("Columna de Profundidad", df_all.columns, index=idx_d)
# ----------------------------------------------

# Limpiar IDs inválidos ("None", "NaN", vacíos)
df_all = df_all[df_all[muestra_col].apply(is_good_id)].copy()

# Forzar conversión numérica de profundidad
if not pd.api.types.is_numeric_dtype(df_all[depth_col]):
    df_all[depth_col] = clean_num(df_all[depth_col])

muestras = sorted(df_all[muestra_col].astype(str).unique().tolist())

with st.sidebar:
    st.markdown("---")
    st.header("🔍 Filtros")
    selected_muestra = st.selectbox("Descripción Muestra", muestras)
    
    use_ensayo_filter = False
    selected_ensayos = []
    if ensayo_col is not None and ensayo_col in df_all.columns:
        all_ensayos = sorted(df_all[ensayo_col].dropna().astype(str).unique().tolist())
        if len(all_ensayos) > 0:
            use_ensayo_filter = st.checkbox("Filtrar por Ensayo", value=False)
            if use_ensayo_filter:
                selected_ensayos = st.multiselect("Ensayo geotecnia", all_ensayos, default=all_ensayos)

# Filtrar el DataFrame principal
sub_df = df_all[df_all[muestra_col].astype(str) == str(selected_muestra)].copy()
sub_df = sub_df.dropna(subset=[depth_col])

if use_ensayo_filter and ensayo_col is not None and len(selected_ensayos) > 0:
    sub_df = sub_df[sub_df[ensayo_col].astype(str).isin([str(x) for x in selected_ensayos])].copy()

# Métricas superiores
k1, k2, k3, k4 = st.columns(4)
k1.metric("Muestra", str(selected_muestra))
k2.metric("Registros", len(sub_df))
k3.metric("Prof min (m)", round(float(sub_df[depth_col].min()), 2) if len(sub_df) else "-")
k4.metric("Prof max (m)", round(float(sub_df[depth_col].max()), 2) if len(sub_df) else "-")

if len(sub_df) == 0:
    st.warning("No hay datos con profundidad para esta muestra (o tras aplicar filtros).")
    st.stop()

# Agrupación y pestañas
groups = group_numeric_columns(sub_df, depth_col)

final_tabs = ["Resumen"]
for t in [
    "SPT y penetración",
    "Plasticidad",
    "Granulometría",
    "Densidad y humedad",
    "Resistencia y corte",
    "Consolidación",
    "Química y otros",
    "Otros"
]:
    if t in groups and any(sub_df[c].notna().sum() > 0 for c in groups[t]):
        final_tabs.append(t)
final_tabs.append("Datos")

tabs = st.tabs(final_tabs)

with tabs[0]:
    st.subheader("Calidad de datos y parámetros disponibles")
    summ_df = numeric_summary_table(sub_df, depth_col)
    st.dataframe(summ_df, use_container_width=True, height=420)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(uscs_pie(sub_df, uscs_col), use_container_width=True)
    with c2:
        if "LL" in sub_df.columns and "IP" in sub_df.columns:
            st.plotly_chart(casagrande_plot(sub_df, "LL", "IP", depth_col=depth_col, uscs_col=uscs_col), use_container_width=True)
        else:
            st.info("No hay columnas LL e IP para la carta de Casagrande en esta selección.")

# Render each group tab dynamically
for idx_tab in range(1, len(final_tabs) - 1):
    tab_name = final_tabs[idx_tab]
    with tabs[idx_tab]:
        st.subheader(tab_name)
        render_group(sub_df, depth_col, groups.get(tab_name, []))

with tabs[-1]:
    st.subheader("Datos filtrados")
    st.dataframe(sub_df.sort_values(depth_col), use_container_width=True, height=520)

    csv_bytes = sub_df.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV (selección)", data=csv_bytes, file_name="geotecnia_filtrado.csv", mime="text/csv")