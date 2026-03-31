
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import unicodedata
from io import BytesIO

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

try:
    import skfuzzy as fuzz
    FUZZY_OK = True
except ImportError:
    FUZZY_OK = False

try:
    import umap
    UMAP_OK = True
except ImportError:
    UMAP_OK = False

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
PALETTE = ["#E07B54","#5B9BD5","#70AD47","#9B59B6","#F1C40F","#1ABC9C","#E74C3C","#2E86AB","#A23B72","#F18F01"]

def norm(s):
    return unicodedata.normalize("NFD", str(s)).encode("ascii","ignore").decode("ascii").lower().strip()

def comma_to_float(val):
    if pd.isna(val): return np.nan
    try: return float(str(val).replace(",",".").strip())
    except: return np.nan

FEATURE_MAP = {
    "Tamiz Grava": "% Grava", "Tamiz Arena": "% Arena", "Tamiz Finos": "% Finos",
    "LL": "LL", "LP": "LP", "IP": "IP",
    "Densidad Seca Kn/m3": "Dens. Seca",
    "Humedad": "Humedad", "RCS (kpa)": "RCS",
    "Angulo de Rozamiento con denaje": "Phi",
    "Cohesion KPa con drenaje": "Cohesion",
    "SPT (valores centrales)": "SPT",
    "MI (valores centrales)": "MI",
    "Indice de Poros": "e0",
    "Clasificacion USCS": "USCS",
    "Unidad geotecnica": "UG_original",
}

USCS_ORDER = {"GP":0,"GW":1,"GM":2,"GC":3,"SP":4,"SW":5,"SM":6,"SC":7,
              "ML":8,"CL":9,"OL":10,"MH":11,"CH":12,"OH":13,"Pt":14}

def load_and_clean(uploaded_file):
    df_raw = pd.read_excel(uploaded_file)
    clean = pd.DataFrame()
    clean["Muestra"] = df_raw.iloc[:,1]
    clean["Ensayo"]  = df_raw.iloc[:,2]
    clean["Prof"]    = pd.to_numeric(df_raw.iloc[:,3], errors="coerce")
    for orig, new in FEATURE_MAP.items():
        col = next((c for c in df_raw.columns if norm(c)==norm(orig)), None)
        if col:
            if new in ["USCS","UG_original","Ensayo"]:
                clean[new] = df_raw[col]
            else:
                clean[new] = df_raw[col].apply(comma_to_float)
    clean["n_valid"] = clean.notna().sum(axis=1)
    clean = clean.sort_values("n_valid", ascending=False)
    clean = clean.drop_duplicates(subset=["Muestra","Prof"], keep="first")
    clean = clean.sort_values(["Muestra","Prof"]).drop(columns="n_valid").reset_index(drop=True)
    return clean

def preprocess(df, feat_cols, n_neighbors=5):
    X = df[feat_cols].copy()
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_imp)
    return X_imp, X_sc, scaler

def run_pca(X_sc):
    pca = PCA()
    pca.fit(X_sc)
    return pca

def run_ward(X_sc):
    Z = linkage(X_sc, method="ward")
    return Z

def run_fuzzy(X_sc, K, m=2.0, error=1e-5, maxiter=1000):
    if not FUZZY_OK:
        return None, None
    data = X_sc.T
    cntr, u, *_ = fuzz.cmeans(data, K, m, error=error, maxiter=maxiter, init=None)
    labels = np.argmax(u, axis=0)
    return labels, u

def fuzzy_partition_coeff(u):
    n = u.shape[1]
    return np.sum(u**2) / n

def metrics_for_k(X_sc, labels):
    if len(np.unique(labels)) < 2:
        return np.nan, np.nan, np.nan
    s = silhouette_score(X_sc, labels)
    ch = calinski_harabasz_score(X_sc, labels)
    db = davies_bouldin_score(X_sc, labels)
    return s, ch, db



def tab_datos(df, feat_cols):
    st.subheader("Vista previa del dataset")
    col1, col2, col3 = st.columns(3)
    col1.metric("Muestras totales", len(df))
    col2.metric("Variables disponibles", len(feat_cols))
    col3.metric("Sondeos", df["Muestra"].nunique())
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Disponibilidad de datos por variable**")
        avail = (100 - df[feat_cols].isna().mean()*100).sort_values(ascending=True)
        colors_bar = ["#E07B54" if v < 20 else "#F1C40F" if v < 50 else "#70AD47" for v in avail.values]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(avail.index, avail.values, color=colors_bar, edgecolor="white")
        ax.axvline(50, color="red", ls="--", lw=1)
        ax.set_xlabel("% datos disponibles")
        ax.set_xlim(0,115)
        for i, v in enumerate(avail.values):
            ax.text(v+1, i, str(round(v,1))+"%", va="center", fontsize=7)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.markdown("**Distribucion de tipos de suelo (USCS)**")
        uscs = df["USCS"].dropna().value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(uscs.index, uscs.values,
                      color=[PALETTE[i % len(PALETTE)] for i in range(len(uscs))],
                      edgecolor="white")
        ax.set_xlabel("Clasificacion USCS")
        ax.set_ylabel("N muestras")
        ax.tick_params(axis="x", rotation=45)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                    str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()


def tab_preprocesamiento(df, feat_cols):
    st.subheader("Preprocesamiento de datos")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Selecciona variables para el clustering**")
        selected = st.multiselect("Variables", feat_cols, default=[f for f in feat_cols if f != "e0"])
    with col2:
        k_knn = st.slider("Vecinos KNN para imputacion", 2, 10, 5)

    if not selected:
        st.warning("Selecciona al menos 2 variables")
        return None, None, None, selected

    X_imp, X_sc, scaler = preprocess(df, selected, k_knn)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Heatmap de correlacion (datos imputados)**")
        corr = pd.DataFrame(X_imp, columns=selected).corr()
        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(selected))); ax.set_xticklabels(selected, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(selected))); ax.set_yticklabels(selected, fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(len(selected)):
            for j in range(len(selected)):
                ax.text(j, i, str(round(corr.values[i,j],2)),
                        ha="center", va="center", fontsize=6,
                        color="white" if abs(corr.values[i,j]) > 0.7 else "black")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.markdown("**Distribucion normalizada por variable**")
        df_sc = pd.DataFrame(X_sc, columns=selected)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.boxplot([df_sc[c].values for c in selected], labels=selected, patch_artist=True,
                   boxprops=dict(facecolor="#5B9BD5", alpha=0.6))
        ax.axhline(0, color="red", ls="--", lw=1)
        ax.set_ylabel("Valor normalizado (z-score)")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    return X_imp, X_sc, scaler, selected



def tab_pca_umap(X_sc, df, selected):
    st.subheader("Reduccion Dimensional")

    pca = run_pca(X_sc)
    var_exp = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(var_exp)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Varianza explicada por componente (PCA)**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(range(1, len(var_exp)+1), var_exp, color="#5B9BD5", alpha=0.8, label="Varianza individual")
        ax2 = ax.twinx()
        ax2.plot(range(1, len(cum_var)+1), cum_var, "o-", color="#E07B54", label="Acumulada")
        ax2.axhline(80, color="green", ls="--", lw=1)
        ax.set_xlabel("Componente"); ax.set_ylabel("% varianza"); ax2.set_ylabel("% acumulada")
        ax.set_title("Scree Plot PCA")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.markdown("**Biplot PCA (PC1 vs PC2)**")
        pca2 = PCA(n_components=2)
        X_pca = pca2.fit_transform(X_sc)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        uscs_vals = df["USCS"].fillna("N/A").values
        uscs_unique = sorted(set(uscs_vals))
        cmap_u = {u: PALETTE[i % len(PALETTE)] for i, u in enumerate(uscs_unique)}
        for u in uscs_unique:
            mask = uscs_vals == u
            ax.scatter(X_pca[mask,0], X_pca[mask,1], c=cmap_u[u], label=u, s=40, alpha=0.7)
        # cargas (loadings)
        loadings = pca2.components_.T
        for i, feat in enumerate(selected):
            ax.arrow(0, 0, loadings[i,0]*2, loadings[i,1]*2, color="gray", alpha=0.5, head_width=0.08)
            ax.text(loadings[i,0]*2.2, loadings[i,1]*2.2, feat, fontsize=7, color="gray")
        ax.set_xlabel("PC1 (" + str(round(pca2.explained_variance_ratio_[0]*100,1)) + "%)")
        ax.set_ylabel("PC2 (" + str(round(pca2.explained_variance_ratio_[1]*100,1)) + "%)")
        ax.axhline(0, color="lightgray", lw=0.5); ax.axvline(0, color="lightgray", lw=0.5)
        ax.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    if UMAP_OK:
        st.markdown("**UMAP 2D**")
        n_n = st.slider("UMAP n_neighbors", 5, 30, 15)
        reducer = umap.UMAP(n_neighbors=n_n, random_state=42)
        X_umap = reducer.fit_transform(X_sc)
        fig, ax = plt.subplots(figsize=(7, 4))
        for u in uscs_unique:
            mask = uscs_vals == u
            ax.scatter(X_umap[mask,0], X_umap[mask,1], c=cmap_u[u], label=u, s=50, alpha=0.75)
        ax.legend(fontsize=7, ncol=3)
        ax.set_title("UMAP 2D - coloreado por USCS")
        plt.tight_layout()
        st.pyplot(fig); plt.close()
    else:
        st.info("Instala umap-learn para activar UMAP: pip install umap-learn")

    return pca


def tab_ward(X_sc, df, selected):
    st.subheader("Clustering Jerarquico Ward")
    Z = run_ward(X_sc)

    col_a, col_b = st.columns([2,1])
    with col_a:
        st.markdown("**Dendrograma Ward**")
        K_line = st.slider("Linea de corte (num. clusters K)", 2, 10, 4)
        fig, ax = plt.subplots(figsize=(10, 4))
        dn = dendrogram(Z, ax=ax, no_labels=True, color_threshold=None,
                        above_threshold_color="gray")
        heights = sorted([d[2] for d in Z], reverse=True)
        if K_line <= len(heights):
            cut_h = (heights[K_line-2] + heights[K_line-1]) / 2
            ax.axhline(cut_h, color="red", ls="--", lw=1.5, label="Corte K="+str(K_line))
            ax.legend(fontsize=8)
        ax.set_ylabel("Distancia Ward")
        ax.set_xlabel("Muestras")
        ax.set_title("Dendrograma - Clustering Jerarquico Ward")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.markdown("**Metricas de validacion vs K**")
        k_range = range(2, min(11, len(X_sc)))
        sil_vals, ch_vals, db_vals = [], [], []
        for k in k_range:
            lbls = fcluster(Z, k, criterion="maxclust") - 1
            s, ch, db = metrics_for_k(X_sc, lbls)
            sil_vals.append(s); ch_vals.append(ch); db_vals.append(db)
        fig, axes = plt.subplots(3, 1, figsize=(4, 5.5))
        for ax_i, (vals, title, color) in zip(axes, [
                (sil_vals,"Silhouette (max mejor)","#5B9BD5"),
                (ch_vals,"Calinski-Harabasz (max mejor)","#70AD47"),
                (db_vals,"Davies-Bouldin (min mejor)","#E07B54")]):
            ax_i.plot(list(k_range), vals, "o-", color=color)
            ax_i.set_title(title, fontsize=7)
            ax_i.set_xlabel("K", fontsize=7)
            ax_i.tick_params(labelsize=7)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    labels_ward = fcluster(Z, K_line, criterion="maxclust") - 1
    pca2 = PCA(n_components=2)
    X_pca = pca2.fit_transform(X_sc)
    st.markdown("**Clusters Ward en espacio PCA 2D**")
    fig, ax = plt.subplots(figsize=(7, 4))
    for k in range(K_line):
        mask = labels_ward == k
        ax.scatter(X_pca[mask,0], X_pca[mask,1], c=PALETTE[k % len(PALETTE)],
                   label="UG-"+str(k+1), s=60, alpha=0.8)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Clusters Ward K=" + str(K_line) + " en PCA")
    ax.legend(); plt.tight_layout()
    st.pyplot(fig); plt.close()

    return labels_ward, K_line, Z


def tab_fuzzy(X_sc, df, K_ward, selected):
    st.subheader("Fuzzy C-Means Clustering")

    if not FUZZY_OK:
        st.error("Instala scikit-fuzzy: pip install scikit-fuzzy")
        return None, None

    K = st.slider("Numero de clusters K (Fuzzy)", 2, 10, K_ward)
    m = st.slider("Factor de fuzziness m (1=crisp, 3=muy difuso)", 1.1, 3.0, 2.0, 0.1)

    labels_fuzzy, u_matrix = run_fuzzy(X_sc, K, m)
    if labels_fuzzy is None:
        st.error("Error en Fuzzy C-Means")
        return None, None

    col_a, col_b = st.columns(2)
    pca2 = PCA(n_components=2)
    X_pca = pca2.fit_transform(X_sc)

    with col_a:
        st.markdown("**Clusters Fuzzy en PCA 2D**")
        fig, ax = plt.subplots(figsize=(5, 4))
        for k in range(K):
            mask = labels_fuzzy == k
            ax.scatter(X_pca[mask,0], X_pca[mask,1], c=PALETTE[k % len(PALETTE)],
                       label="UG-"+str(k+1), s=60, alpha=0.8)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title("Clusters Fuzzy K=" + str(K))
        ax.legend(fontsize=8); plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.markdown("**Matriz de membresia (primeras 25 muestras)**")
        u_df = pd.DataFrame(u_matrix.T, columns=["UG-"+str(i+1) for i in range(K)])
        u_df["Muestra"] = df["Muestra"].values
        u_df["Prof"] = df["Prof"].values
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(u_df.iloc[:25, :K].values.T, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_yticks(range(K)); ax.set_yticklabels(["UG-"+str(i+1) for i in range(K)])
        ax.set_xlabel("Muestra (primeras 25)")
        ax.set_title("Grados de membresia Fuzzy")
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    return labels_fuzzy, u_matrix



def tab_validacion(X_sc, labels_ward, labels_fuzzy, K_ward):
    st.subheader("Validacion del Clustering")

    methods = {}
    if labels_ward is not None:
        methods["Ward (K="+str(K_ward)+")"] = labels_ward
    if labels_fuzzy is not None:
        methods["Fuzzy C-Means"] = labels_fuzzy

    if not methods:
        st.warning("Ejecuta primero Ward y Fuzzy")
        return

    rows = []
    for name, lbls in methods.items():
        s, ch, db = metrics_for_k(X_sc, lbls)
        rows.append({"Metodo": name, "Silhouette": round(s,3),
                     "Calinski-Harabasz": round(ch,1), "Davies-Bouldin": round(db,3)})
    df_met = pd.DataFrame(rows)
    st.dataframe(df_met, use_container_width=True)

    col_a, col_b, col_c = st.columns(3)
    metrics_info = [
        ("Silhouette", "Silhouette Score", "#5B9BD5", "max"),
        ("Calinski-Harabasz", "Calinski-Harabasz", "#70AD47", "max"),
        ("Davies-Bouldin", "Davies-Bouldin", "#E07B54", "min"),
    ]
    for col, (key, title, color, best) in zip([col_a, col_b, col_c], metrics_info):
        with col:
            fig, ax = plt.subplots(figsize=(3.5, 2.5))
            ax.bar(df_met["Metodo"], df_met[key], color=color, alpha=0.8, edgecolor="white")
            ax.set_title(title + " (" + best + " mejor)", fontsize=8)
            ax.tick_params(axis="x", rotation=15, labelsize=7)
            plt.tight_layout(); st.pyplot(fig); plt.close()


def tab_resultados(df, labels_ward, labels_fuzzy, u_matrix, feat_cols_used, K_ward):
    st.subheader("Resultados por Unidad Geotecnica (UG)")

    labels = labels_ward if labels_ward is not None else labels_fuzzy
    if labels is None:
        st.warning("Ejecuta primero el clustering")
        return df

    df_res = df.copy()
    df_res["UG_Ward"] = ["UG-"+str(l+1) for l in labels]
    if labels_fuzzy is not None:
        df_res["UG_Fuzzy"] = ["UG-"+str(l+1) for l in labels_fuzzy]
        if u_matrix is not None:
            K = u_matrix.shape[0]
            for k in range(K):
                df_res["Memb_UG-"+str(k+1)] = u_matrix[k]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Tabla de parametros por UG (Ward)**")
        num_cols = [c for c in feat_cols_used if c in df_res.columns]
        tbl = df_res.groupby("UG_Ward")[num_cols].agg(["mean","std","count"]).round(2)
        st.dataframe(tbl, use_container_width=True)

    with col_b:
        st.markdown("**Carta de Plasticidad (Casagrande) por UG**")
        if "LL" in df_res.columns and "IP" in df_res.columns:
            fig, ax = plt.subplots(figsize=(5, 4))
            ll_r = np.linspace(0, 120, 200)
            ax.plot(ll_r, 0.73*(ll_r-20), "k--", lw=1, label="Linea A")
            ax.axvline(50, color="gray", lw=0.8)
            for i, ug in enumerate(sorted(df_res["UG_Ward"].unique())):
                sub = df_res[df_res["UG_Ward"]==ug].dropna(subset=["LL","IP"])
                ax.scatter(sub["LL"], sub["IP"], c=PALETTE[i % len(PALETTE)],
                           label=ug, s=60, alpha=0.8, zorder=3)
            ax.set_xlabel("LL"); ax.set_ylabel("IP")
            ax.set_xlim(0,110); ax.set_ylim(0,80)
            ax.legend(fontsize=7); ax.set_title("Casagrande por UG")
            plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("**Columnas estratigraficas por sondeo**")
    sondeos = sorted(df_res["Muestra"].dropna().unique())
    n_cols = st.slider("Sondeos por fila", 2, 8, 4)
    cols = st.columns(n_cols)
    for idx, sond in enumerate(sondeos):
        sub = df_res[df_res["Muestra"]==sond].sort_values("Prof")
        if sub.empty: continue
        with cols[idx % n_cols]:
            fig, ax = plt.subplots(figsize=(1.4, 3.5))
            prof_max = sub["Prof"].max() if sub["Prof"].max() > 0 else 1
            ug_unique = sorted(df_res["UG_Ward"].unique())
            ug_cmap = {ug: PALETTE[i % len(PALETTE)] for i, ug in enumerate(ug_unique)}
            for _, row in sub.iterrows():
                height = row["Prof"] if idx==0 else (sub.iloc[list(sub.index).index(row.name)-1]["Prof"]
                    if list(sub.index).index(row.name)>0 else row["Prof"])
                rect = plt.Rectangle((0, prof_max - row["Prof"]),
                                      1, row["Prof"]/len(sub),
                                      facecolor=ug_cmap.get(row.get("UG_Ward","UG-1"), PALETTE[0]),
                                      edgecolor="white", lw=0.5)
                ax.add_patch(rect)
                ax.text(0.5, prof_max - row["Prof"] + row["Prof"]/(2*len(sub)),
                        row.get("UG_Ward","?"), ha="center", va="center", fontsize=5, fontweight="bold")
            ax.set_xlim(0,1); ax.set_ylim(0, prof_max)
            ax.set_title(str(sond), fontsize=6)
            ax.axis("off")
            plt.tight_layout(); st.pyplot(fig); plt.close()

    return df_res


def tab_exportar(df_res):
    st.subheader("Exportar resultados")
    st.markdown("Descarga el dataset completo con las UG asignadas.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Vista previa resultados**")
        cols_show = [c for c in ["Muestra","Prof","USCS","UG_Ward","UG_Fuzzy"] if c in df_res.columns]
        st.dataframe(df_res[cols_show].head(30), use_container_width=True)

    with col2:
        st.markdown("**Estadisticas por UG**")
        num_cols = df_res.select_dtypes(include=np.number).columns.tolist()
        num_cols = [c for c in num_cols if not c.startswith("Memb_")]
        if "UG_Ward" in df_res.columns:
            st.dataframe(df_res.groupby("UG_Ward")[num_cols].mean().round(2), use_container_width=True)

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_res.to_excel(writer, sheet_name="Datos_con_UG", index=False)
        if "UG_Ward" in df_res.columns:
            num_cols2 = df_res.select_dtypes(include=np.number).columns.tolist()
            num_cols2 = [c for c in num_cols2 if not c.startswith("Memb_")]
            df_res.groupby("UG_Ward")[num_cols2].mean().round(2).to_excel(writer, sheet_name="Parametros_UG")
    buffer.seek(0)
    st.download_button("Descargar Excel con UG asignadas",
                       data=buffer, file_name="resultados_geotecnia_UG.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")



def main():
    st.set_page_config(page_title="Clasificacion Geotecnica", layout="wide",
                       page_icon="🪨")
    st.title("🪨 Clasificacion Geotecnica Automatizada")
    st.markdown("Carga un archivo Excel con datos de laboratorio y sondeos SPT/MI para obtener Unidades Geotecnicas (UG) por clustering.")

    # ── Sidebar ──
    with st.sidebar:
        st.header("Configuracion")
        uploaded = st.file_uploader("Sube tu archivo Excel (.xlsx)", type=["xlsx","xls"])
        st.markdown("---")
        st.caption("Metodos: Ward Jerarquico + Fuzzy C-Means + GMM")
        st.caption("v2.0 | Julius AI")

    if uploaded is None:
        st.info("Sube un archivo Excel con datos geotecnicos para comenzar.")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tab 1", "Datos")
        col2.metric("Tab 2", "Preprocesamiento")
        col3.metric("Tab 3", "PCA / UMAP")
        col4.metric("Tab 4", "Ward Jerarquico")
        col1b, col2b, col3b, col4b = st.columns(4)
        col1b.metric("Tab 5", "Fuzzy C-Means")
        col2b.metric("Tab 6", "Validacion")
        col3b.metric("Tab 7", "Resultados & UG")
        col4b.metric("Tab 8", "Exportar")
        return

    df = load_and_clean(uploaded)

    NUMERIC_FEATS = ["% Grava","% Arena","% Finos","LL","LP","IP",
                     "Dens. Seca","Humedad","RCS","Phi","Cohesion","SPT","MI","e0"]
    feat_cols = [f for f in NUMERIC_FEATS if f in df.columns and df[f].notna().sum() > 5]

    # ── Persistencia en session_state ──
    if "X_sc" not in st.session_state: st.session_state.X_sc = None
    if "labels_ward" not in st.session_state: st.session_state.labels_ward = None
    if "labels_fuzzy" not in st.session_state: st.session_state.labels_fuzzy = None
    if "u_matrix" not in st.session_state: st.session_state.u_matrix = None
    if "K_ward" not in st.session_state: st.session_state.K_ward = 4
    if "df_res" not in st.session_state: st.session_state.df_res = df.copy()
    if "feat_used" not in st.session_state: st.session_state.feat_used = feat_cols

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📂 Datos",
        "🔬 Preprocesamiento",
        "📉 PCA / UMAP",
        "🌿 Ward Jerarquico",
        "🔵 Fuzzy C-Means",
        "🎯 Validacion",
        "📊 Resultados & UG",
        "💾 Exportar",
    ])

    with tab1:
        tab_datos(df, feat_cols)

    with tab2:
        X_imp, X_sc, scaler, selected = tab_preprocesamiento(df, feat_cols)
        if X_sc is not None:
            st.session_state.X_sc = X_sc
            st.session_state.feat_used = selected

    X_sc = st.session_state.X_sc
    if X_sc is None:
        for t in [tab3, tab4, tab5, tab6, tab7, tab8]:
            with t:
                st.warning("Completa la Etapa de Preprocesamiento primero (Tab 2).")
        return

    with tab3:
        tab_pca_umap(X_sc, df, st.session_state.feat_used)

    with tab4:
        labels_ward, K_ward, Z = tab_ward(X_sc, df, st.session_state.feat_used)
        st.session_state.labels_ward = labels_ward
        st.session_state.K_ward = K_ward

    with tab5:
        labels_fuzzy, u_matrix = tab_fuzzy(X_sc, df, st.session_state.K_ward, st.session_state.feat_used)
        st.session_state.labels_fuzzy = labels_fuzzy
        st.session_state.u_matrix = u_matrix

    with tab6:
        tab_validacion(X_sc, st.session_state.labels_ward, st.session_state.labels_fuzzy, st.session_state.K_ward)

    with tab7:
        df_res = tab_resultados(df, st.session_state.labels_ward, st.session_state.labels_fuzzy,
                                st.session_state.u_matrix, st.session_state.feat_used, st.session_state.K_ward)
        st.session_state.df_res = df_res

    with tab8:
        tab_exportar(st.session_state.df_res)


if __name__ == "__main__":
    main()


