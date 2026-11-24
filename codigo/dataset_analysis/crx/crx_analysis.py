"""Análisis exploratorio del dataset Credit Approval (UCI, crx) seguido de su limpieza.

Ejecución: python crx_analysis.py; genera las figuras en img/ y, con cleanup.py, escribe crx_cleaned.csv.
"""

# === 0. Configuración ===
import importlib.util
import sys
from pathlib import Path

import matplotlib

# Usamos el backend Agg para poder ejecutar el script sin entorno gráfico.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

# La limpieza vive solo en cleanup.py: la cargamos por ruta (sin tocar sys.path) y sin escribir __pycache__.
sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("crx_cleanup", Path(__file__).resolve().parent / "cleanup.py")
crx_cleanup = importlib.util.module_from_spec(_spec)
sys.modules["crx_cleanup"] = crx_cleanup
_spec.loader.exec_module(crx_cleanup)
run_cleanup, section = crx_cleanup.run_cleanup, crx_cleanup.section

SEED = 42
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_PATH = SCRIPT_DIR / ".." / ".." / "datasets" / "crx.csv"
IMG_DIR = SCRIPT_DIR / "img"
DPI = 200

COLUMNS = [f"A{i}" for i in range(1, 17)]
TARGET = "A16"

# Esquema documentado en crx.names (UCI): niveles permitidos por variable categórica.
DOCUMENTED_LEVELS = {
    "A1": ["b", "a"],
    "A4": ["u", "y", "l", "t"],
    "A5": ["g", "p", "gg"],
    "A6": ["c", "d", "cc", "i", "j", "k", "m", "r", "q", "w", "x", "e", "aa", "ff"],
    "A7": ["v", "h", "bb", "j", "n", "z", "dd", "ff", "o"],
    "A9": ["t", "f"],
    "A10": ["t", "f"],
    "A12": ["t", "f"],
    "A13": ["g", "p", "s"],
    "A16": ["+", "-"],
}
CONTINUOUS = ["A2", "A3", "A8", "A11", "A14", "A15"]
CATEGORICAL = [c for c in DOCUMENTED_LEVELS if c != TARGET]
DOCUMENTED_MISSING = {"A1": 12, "A2": 12, "A4": 6, "A5": 6, "A6": 9, "A7": 9, "A14": 13}
DOCUMENTED_CLASSES = {"+": 307, "-": 383}

# Colores: azul para aprobada (+) y naranja para denegada (-), fijos en todas las figuras.
COLOR_POS = "#2a78d6"
COLOR_NEG = "#eb6834"
CLASS_COLORS = {"+": COLOR_POS, "-": COLOR_NEG}
CLASS_LABELS = {"+": "Aprobada (+)", "-": "Denegada (-)"}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": INK_SECONDARY,
    "axes.labelcolor": INK,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "font.size": 9,
    "legend.frameon": False,
})


def save_figure(fig, name, description):
    """Guardamos la figura en img/, la cerramos e imprimimos la traza."""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG_DIR / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f'Graph "{description}" saved to img/{name}.png')


def holm_correction(p_values):
    """Corrección de Holm-Bonferroni (controla FWER sin suponer independencia)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def cramers_v(x, y):
    """V de Cramér con corrección de sesgo de Bergsma (2013)."""
    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    r, k = table.shape
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return np.sqrt(phi2_corr / denom) if denom > 0 else np.nan


def correlation_ratio(categories, values):
    """Razón de correlación eta (sobre rangos) entre una categórica y una continua."""
    mask = categories.notna() & values.notna()
    # Usamos rangos para que eta sea robusta a las colas largas, igual que Spearman.
    cats, vals = categories[mask], values[mask].astype(float).rank()
    grand_mean = vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for _, g in vals.groupby(cats))
    ss_total = ((vals - grand_mean) ** 2).sum()
    return np.sqrt(ss_between / ss_total) if ss_total > 0 else np.nan


def chi2_permutation_pvalue(x, y, n_permutations=9999, seed=SEED):
    """p-valor de chi-cuadrado por permutación, válido con frecuencias esperadas bajas."""
    rng = np.random.default_rng(seed)
    x_codes = pd.factorize(x)[0]
    y_codes = pd.factorize(y)[0]
    observed = stats.chi2_contingency(pd.crosstab(x_codes, y_codes), correction=False)[0]
    count = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(y_codes)
        if stats.chi2_contingency(pd.crosstab(x_codes, permuted), correction=False)[0] >= observed:
            count += 1
    return (count + 1) / (n_permutations + 1)


# ============================================================================
# PARTE 1: ANÁLISIS
# ============================================================================

def load_for_analysis():
    # === 1. Carga y procedencia ===
    section("1. Carga y procedencia")
    # Leemos sin cabecera porque crx.csv es una copia de crx.data de UCI, que no la tiene.
    raw = pd.read_csv(RAW_PATH, header=None, names=COLUMNS, dtype=str)
    print(f"Fichero: {RAW_PATH.resolve()}")
    print(f"Forma cruda: {raw.shape} (documentado en crx.names: 690 x 16)")
    assert raw.shape == (690, 16), "El fichero no coincide con la documentación de UCI"
    question_marks = (raw == "?").sum()
    print(f"Marcadores '?' totales: {int(question_marks.sum())}")
    # Convertimos los `?` a NaN porque crx.names los define como valor ausente.
    df = raw.replace("?", np.nan)
    for col in CONTINUOUS:
        df[col] = pd.to_numeric(df[col], errors="raise")
    print("\nPrimeras filas:")
    print(df.head().to_string())
    return raw, df


def analyse_schema(raw, df):
    # === 2. Esquema y tipos ===
    section("2. Esquema y tipos")
    # Comprobamos que cada columna continua se parsea como número y vemos si es entera.
    rows = []
    for col in COLUMNS:
        kind = "continua" if col in CONTINUOUS else ("objetivo" if col == TARGET else "categórica")
        values = df[col].dropna()
        is_integer = bool(kind == "continua" and np.all(np.mod(values, 1) == 0))
        rows.append({"columna": col, "tipo_doc": kind, "dtype": str(df[col].dtype),
                     "n_unicos": values.nunique(), "entera": is_integer if kind == "continua" else "-"})
    schema = pd.DataFrame(rows).set_index("columna")
    print(schema.to_string())
    # Buscamos niveles no documentados y raros (<1 % de filas) porque generan celdas casi vacías.
    print("\nNiveles observados frente a documentados:")
    rare_threshold = 0.01 * len(df)
    for col in DOCUMENTED_LEVELS:
        observed = set(df[col].dropna().unique())
        documented = set(DOCUMENTED_LEVELS[col])
        undocumented = sorted(observed - documented)
        unused = sorted(documented - observed)
        counts = df[col].value_counts()
        rare = counts[counts < rare_threshold].to_dict()
        print(f"  {col}: no documentados={undocumented or 'ninguno'}; "
              f"documentados sin uso={unused or 'ninguno'}; raros(<{rare_threshold:.1f})={rare or 'ninguno'}")
    # Buscamos espacios o mayúsculas extrañas en las categóricas, típicas erratas de transcripción.
    odd = {c: int((raw[c].str.strip() != raw[c]).sum() + (raw[c].str.lower() != raw[c]).sum())
           for c in DOCUMENTED_LEVELS}
    print(f"\nValores con espacios o mayúsculas: {odd}")
    return schema


def analyse_missing(df):
    # === 3. Valores ausentes ===
    section("3. Valores ausentes")
    missing_counts = df.isna().sum()
    documented = pd.Series(DOCUMENTED_MISSING).reindex(COLUMNS).fillna(0).astype(int)
    table = pd.DataFrame({"observados": missing_counts, "documentados": documented,
                          "%": (100 * missing_counts / len(df)).round(2)})
    print(table[table.observados > 0].to_string())
    print(f"Coincide con crx.names: {bool((missing_counts == documented).all())}")
    row_missing = df.isna().sum(axis=1)
    rows_with_missing = int((row_missing > 0).sum())
    print(f"Filas con algún ausente: {rows_with_missing} ({100 * rows_with_missing / len(df):.1f} %)"
          f" | filas completas: {len(df) - rows_with_missing}")
    print("Distribución de nº de ausentes por fila:", row_missing.value_counts().sort_index().to_dict())

    # Contamos la co-ocurrencia porque un bloque de columnas ausentes juntas sugiere un mecanismo común.
    indicator = df.isna().astype(int)
    missing_cols = [c for c in COLUMNS if missing_counts[c] > 0]
    co_occurrence = indicator[missing_cols].T.dot(indicator[missing_cols])
    print("\nCo-ocurrencia de ausentes:")
    print(co_occurrence.to_string())
    patterns = indicator[missing_cols][row_missing > 0].astype(str).agg("".join, axis=1)
    pattern_names = patterns.map(lambda s: "+".join(c for c, b in zip(missing_cols, s) if b == "1"))
    print("Patrones de ausencia:", pattern_names.value_counts().to_dict())

    # Contrastamos ausencia vs objetivo con Fisher exacto porque las tablas 2x2 tienen celdas pequeñas.
    print("\nAusencia frente a objetivo (Fisher exacto, bilateral):")
    any_missing = row_missing > 0
    tests = {"cualquier columna": any_missing}
    tests.update({c: df[c].isna() for c in missing_cols})
    for name, mask in tests.items():
        contingency = pd.crosstab(mask, df[TARGET]).reindex(index=[False, True], columns=["+", "-"], fill_value=0)
        odds, p = stats.fisher_exact(contingency)
        rate = 100 * (df.loc[mask, TARGET] == "-").mean()
        print(f"  {name:>18}: n={int(mask.sum()):2d}, % denegadas en ausentes={rate:5.1f} "
              f"(global {100 * (df[TARGET] == '-').mean():.1f}), OR={odds:.2f}, p={p:.4f}")
    # Comparamos las continuas completas entre filas con y sin ausentes (Mann-Whitney) para razonar MCAR/MAR.
    print("\nContinuas completas: filas con ausentes frente a filas completas (Mann-Whitney U):")
    for col in [c for c in CONTINUOUS if missing_counts[c] == 0]:
        u, p = stats.mannwhitneyu(df.loc[any_missing, col], df.loc[~any_missing, col])
        print(f"  {col}: mediana ausentes={df.loc[any_missing, col].median():.3f}, "
              f"completas={df.loc[~any_missing, col].median():.3f}, p={p:.4f}")

    # Figura: barras de ausentes por columna y mapa de ausencias por fila.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    bars = ax.bar(COLUMNS, missing_counts.values, color=COLOR_POS, width=0.6)
    for bar, value in zip(bars, missing_counts.values):
        if value > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.2, str(value),
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_title("Valores ausentes por columna")
    ax.set_ylabel("Nº de filas con '?'")
    ax.tick_params(axis="x", rotation=45)
    ax = axes[1]
    order = np.argsort(-row_missing.values, kind="stable")
    sub = indicator.iloc[order[:rows_with_missing]]
    ax.imshow(sub.values, aspect="auto", cmap=ListedColormap(["#f0efec", COLOR_NEG]), interpolation="nearest")
    ax.set_xticks(range(len(COLUMNS)), COLUMNS, rotation=45)
    ax.set_yticks([0, rows_with_missing - 1], ["1", str(rows_with_missing)])
    ax.set_ylabel("Filas con algún ausente (ordenadas)")
    ax.set_title(f"Mapa de ausencias ({rows_with_missing} de {len(df)} filas)")
    ax.grid(False)
    fig.suptitle("Credit Approval: valores ausentes", fontweight="bold")
    save_figure(fig, "missing_values", "missing values per column and missingness map")

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(co_occurrence.values, cmap="Blues")
    ax.set_xticks(range(len(missing_cols)), missing_cols)
    ax.set_yticks(range(len(missing_cols)), missing_cols)
    for i in range(len(missing_cols)):
        for j in range(len(missing_cols)):
            value = co_occurrence.values[i, j]
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8,
                    color="white" if value > co_occurrence.values.max() / 2 else INK)
    ax.set_title("Co-ocurrencia de ausentes\n(diagonal = ausentes por columna)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Nº de filas")
    save_figure(fig, "missing_cooccurrence", "co-occurrence of missing values between columns")


def analyse_validity(df):
    # === 4. Valores inválidos y consistencia ===
    section("4. Valores inválidos y consistencia")
    # Revisamos rangos porque las continuas anonimizadas no deberían ser negativas (cantidades o recuentos).
    for col in CONTINUOUS:
        values = df[col].dropna()
        print(f"  {col}: min={values.min()}, max={values.max()}, negativos={int((values < 0).sum())}, "
              f"ceros={int((values == 0).sum())} ({100 * (values == 0).mean():.1f} %)")
    # Comprobamos si A4 y A5 son una recodificación una de otra, porque eso las haría redundantes.
    a4_a5 = pd.crosstab(df["A4"], df["A5"])
    print("\nTabla A4 x A5:")
    print(a4_a5.to_string())
    bijective = bool(((a4_a5 > 0).sum(axis=0) == 1).all() and ((a4_a5 > 0).sum(axis=1) == 1).all())
    print(f"A4 <-> A5 biyectiva: {bijective}")
    # Comprobamos si A10 es el indicador A11 > 0, porque sería información derivada.
    both = df[["A10", "A11"]].dropna()
    consistency = ((both["A11"] > 0) == (both["A10"] == "t")).mean()
    print(f"A10 == 't' si y solo si A11 > 0: {100 * consistency:.1f} % de las filas")
    # Miramos la coincidencia del nivel 'ff' en A6 y A7 porque sugiere una codificación compartida.
    print("\nA6 = 'ff' frente a A7:", df.loc[df["A6"] == "ff", "A7"].value_counts().to_dict())
    print("A7 = 'ff' frente a A6:", df.loc[df["A7"] == "ff", "A6"].value_counts().to_dict())
    print("A6 = 'j' frente a A7:", df.loc[df["A6"] == "j", "A7"].value_counts().to_dict())
    # A14 venía con ceros a la izquierda (5 dígitos) en crx.data, lo que sugiere un código y no una magnitud.
    print(f"\nA14: {df['A14'].nunique()} valores distintos; los 8 más frecuentes: "
          f"{df['A14'].value_counts().head(8).to_dict()}")
    print(f"A14 múltiplo de 10: {100 * (df['A14'].dropna() % 10 == 0).mean():.1f} %")
    return bijective


def analyse_duplicates_cardinality(df):
    # === 5. Duplicados, columnas casi constantes y cardinalidad ===
    section("5. Duplicados, columnas casi constantes y cardinalidad")
    # Buscamos duplicados con y sin el objetivo porque los segundos revelarían etiquetas contradictorias.
    features = [c for c in COLUMNS if c != TARGET]
    print(f"Filas duplicadas exactas: {int(df.duplicated().sum())}")
    print(f"Filas duplicadas en atributos (ignorando A16): {int(df.duplicated(subset=features).sum())}")
    # Medimos la frecuencia del valor dominante; por encima del 95 % la columna aporta poca variación.
    rows = []
    for col in COLUMNS:
        counts = df[col].value_counts(normalize=True)
        rows.append({"columna": col, "n_unicos": df[col].nunique(),
                     "valor_dominante": counts.index[0], "%_dominante": round(100 * counts.iloc[0], 1)})
    cardinality = pd.DataFrame(rows).set_index("columna")
    print(cardinality.to_string())
    near_constant = cardinality[cardinality["%_dominante"] > 95].index.tolist()
    print(f"Columnas casi constantes (>95 %): {near_constant or 'ninguna'}")


def analyse_continuous(df):
    # === 6. Estadística univariante: continuas ===
    section("6. Estadística univariante: continuas")
    rows = []
    for col in CONTINUOUS:
        x = df[col].dropna()
        q1, q3 = x.quantile([0.25, 0.75])
        rows.append({"col": col, "n": len(x), "media": x.mean(), "mediana": x.median(), "std": x.std(),
                     "min": x.min(), "Q1": q1, "Q3": q3, "IQR": q3 - q1, "max": x.max(),
                     "asimetría": stats.skew(x), "curtosis_exc": stats.kurtosis(x),
                     "asim_log1p": stats.skew(np.log1p(x))})
    summary = pd.DataFrame(rows).set_index("col")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(summary.to_string())

    # Contrastamos normalidad con Shapiro-Wilk (potente para n < 5000) y D'Agostino K² (basado en asimetría y curtosis).
    print("\nNormalidad:")
    for col in CONTINUOUS:
        x = df[col].dropna()
        print(f"  {col}: Shapiro W={stats.shapiro(x).statistic:.3f} p={stats.shapiro(x).pvalue:.2e} | "
              f"D'Agostino K²={stats.normaltest(x).statistic:.1f} p={stats.normaltest(x).pvalue:.2e}")

    # Usamos la regla IQR (y no el z-score) porque las variables son muy asimétricas y la media y la std no son robustas.
    print("\nAtípicos por regla IQR (1.5·IQR) en escala original y en log1p, y |z| > 3 como referencia:")
    for col in CONTINUOUS:
        x = df[col].dropna()
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        iqr_out = int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())
        lx = np.log1p(x)
        lq1, lq3 = lx.quantile([0.25, 0.75])
        log_out = int(((lx < lq1 - 1.5 * (lq3 - lq1)) | (lx > lq3 + 1.5 * (lq3 - lq1))).sum())
        z_out = int((np.abs(stats.zscore(x)) > 3).sum())
        print(f"  {col}: IQR={iqr_out:3d} ({100 * iqr_out / len(x):4.1f} %), IQR(log1p)={log_out:3d}, |z|>3={z_out:3d}")
    print(f"Mayores valores de A15: {sorted(df['A15'].nlargest(5).tolist(), reverse=True)}")

    # Figura: histogramas con KDE en escala original.
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, col in zip(axes.flat, CONTINUOUS):
        x = df[col].dropna()
        ax.hist(x, bins=40, density=True, color=COLOR_POS, alpha=0.55, edgecolor="white", linewidth=0.4)
        grid = np.linspace(x.min(), x.max(), 400)
        ax.plot(grid, stats.gaussian_kde(x)(grid), color=INK, linewidth=1.5, label="KDE")
        ax.axvline(x.median(), color=COLOR_NEG, linewidth=1.5, linestyle="--", label="Mediana")
        ax.set_title(f"{col} (asimetría {stats.skew(x):.2f})")
        ax.set_xlabel(f"{col} (unidades anonimizadas)")
        ax.set_ylabel("Densidad")
    axes.flat[0].legend()
    fig.suptitle("Distribución de las variables continuas (escala original)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "continuous_histograms", "histograms with KDE of continuous columns")

    # Figura: las mismas variables en log1p, porque es la transformación que proponemos para las colas largas.
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, col in zip(axes.flat, CONTINUOUS):
        x = np.log1p(df[col].dropna())
        ax.hist(x, bins=40, density=True, color=COLOR_POS, alpha=0.55, edgecolor="white", linewidth=0.4)
        if x.nunique() > 2:
            grid = np.linspace(x.min(), x.max(), 400)
            ax.plot(grid, stats.gaussian_kde(x)(grid), color=INK, linewidth=1.5)
        ax.set_title(f"log1p({col}) (asimetría {stats.skew(x):.2f})")
        ax.set_xlabel(f"log(1 + {col})")
        ax.set_ylabel("Densidad")
    fig.suptitle("Distribución de las variables continuas tras log1p", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "continuous_histograms_log1p", "histograms with KDE of log1p-transformed continuous columns")
    return summary


def analyse_categorical(df):
    # === 7. Estadística univariante: categóricas ===
    section("7. Estadística univariante: categóricas")
    for col in CATEGORICAL:
        counts = df[col].value_counts(dropna=False)
        pct = (100 * counts / len(df)).round(1)
        print(f"  {col}: " + ", ".join(f"{k}={v} ({p} %)" for k, v, p in zip(counts.index, counts, pct)))

    # Figura: frecuencias por nivel, divididas por clase, con la tasa de aprobación bajo cada nivel.
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, col in zip(axes.flat, CATEGORICAL):
        table = pd.crosstab(df[col], df[TARGET]).reindex(columns=["+", "-"], fill_value=0)
        table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
        positions = np.arange(len(table))
        width = 0.4
        for offset, cls in [(-width / 2, "+"), (width / 2, "-")]:
            ax.bar(positions + offset, table[cls], width=width * 0.95, color=CLASS_COLORS[cls], label=CLASS_LABELS[cls])
        rates = 100 * table["+"] / table.sum(axis=1)
        tick_size = 6 if len(table) > 10 else 7.5
        ax.set_xticks(positions, [f"{lvl}\n{r:.0f}%" for lvl, r in zip(table.index, rates)], fontsize=tick_size)
        ax.set_title(f"{col} (n válidos = {int(table.values.sum())})")
        ax.set_ylabel("Nº de solicitudes")
    axes.flat[0].legend(loc="upper right")
    fig.suptitle("Variables categóricas por clase (bajo cada nivel: % de aprobadas)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "categorical_by_target", "categorical level counts split by target")


def analyse_target(df):
    # === 8. Balance del objetivo ===
    section("8. Balance del objetivo")
    counts = df[TARGET].value_counts().reindex(["+", "-"])
    print(f"Clases: {counts.to_dict()} (documentado {DOCUMENTED_CLASSES})")
    print(f"Proporción: + {100 * counts['+'] / len(df):.1f} % / - {100 * counts['-'] / len(df):.1f} %"
          f" | ratio mayoritaria/minoritaria = {counts.max() / counts.min():.2f}")
    # Con un ratio de 1.25 no aplicamos remuestreo; basta con estratificar las particiones.
    fig, ax = plt.subplots(figsize=(5, 3.6))
    bars = ax.bar([CLASS_LABELS[c] for c in counts.index], counts.values,
                  color=[CLASS_COLORS[c] for c in counts.index], width=0.55)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 5, f"{value} ({100 * value / len(df):.1f} %)",
                ha="center", va="bottom", color=INK)
    ax.set_ylabel("Nº de solicitudes")
    ax.set_title("Balance de la variable objetivo A16")
    ax.set_ylim(0, counts.max() * 1.15)
    save_figure(fig, "target_balance", "target class balance")


def analyse_continuous_vs_target(df):
    # === 9. Bivariante: continuas frente al objetivo ===
    section("9. Bivariante: continuas frente al objetivo")
    # Comprobamos normalidad por grupo: se rechaza en todas, así que usamos Mann-Whitney U y Welch solo como referencia.
    rows = []
    for col in CONTINUOUS:
        pos = df.loc[df[TARGET] == "+", col].dropna()
        neg = df.loc[df[TARGET] == "-", col].dropna()
        normal = stats.shapiro(pos).pvalue > 0.05 and stats.shapiro(neg).pvalue > 0.05
        u, p_mw = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        t, p_t = stats.ttest_ind(pos, neg, equal_var=False)
        # La correlación biserial de rangos es el tamaño de efecto natural de Mann-Whitney.
        rank_biserial = 2 * u / (len(pos) * len(neg)) - 1
        rows.append({"col": col, "mediana_+": pos.median(), "mediana_-": neg.median(),
                     "normal_ambos": normal, "U": u, "p_MW": p_mw, "p_Welch(ref)": p_t,
                     "r_rank_biserial": rank_biserial})
    result = pd.DataFrame(rows).set_index("col")
    result["p_MW_Holm"] = holm_correction(result["p_MW"])
    with pd.option_context("display.float_format", "{:.4g}".format):
        print(result.to_string())
    print("(r > 0: valores mayores en aprobadas; |r| ~0.1 pequeño, ~0.3 medio, ~0.5 grande)")

    # Figura: boxplots por clase en escala log1p para que las colas no aplasten la caja.
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, col in zip(axes.flat, CONTINUOUS):
        data = [np.log1p(df.loc[df[TARGET] == cls, col].dropna()) for cls in ["+", "-"]]
        box = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=True,
                         flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.5, "markeredgecolor": INK_SECONDARY},
                         medianprops={"color": INK, "linewidth": 1.5})
        for patch, cls in zip(box["boxes"], ["+", "-"]):
            patch.set_facecolor(CLASS_COLORS[cls])
            patch.set_alpha(0.75)
        ax.set_xticks([1, 2], [CLASS_LABELS["+"], CLASS_LABELS["-"]])
        ax.set_ylabel(f"log(1 + {col})")
        row = result.loc[col]
        ax.set_title(f"{col}: r = {row['r_rank_biserial']:.2f}, p Holm = {row['p_MW_Holm']:.1e}")
    fig.suptitle("Variables continuas por clase (escala log1p; Mann-Whitney U)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "continuous_boxplots_by_target", "boxplots of log1p continuous columns by target")
    return result


def analyse_categorical_vs_target(df):
    # === 10. Bivariante: categóricas frente al objetivo ===
    section("10. Bivariante: categóricas frente al objetivo")
    # 2x2: Fisher exacto. RxC: chi-cuadrado si todas las esperadas >= 5; si no, p por permutación.
    rows = []
    for col in CATEGORICAL:
        sub = df[[col, TARGET]].dropna()
        table = pd.crosstab(sub[col], sub[TARGET])
        chi2, p_chi2, dof, expected = stats.chi2_contingency(table, correction=False)
        low_expected = int((expected < 5).sum())
        if table.shape == (2, 2):
            test, p = "Fisher exacto", stats.fisher_exact(table)[1]
        elif low_expected == 0:
            test, p = "chi-cuadrado", p_chi2
        else:
            test, p = "chi² permutación", chi2_permutation_pvalue(sub[col], sub[TARGET])
        rows.append({"col": col, "niveles": table.shape[0], "chi2": chi2, "gl": dof,
                     "celdas_esp<5": low_expected, "test": test, "p": p, "V_Cramer": cramers_v(sub[col], sub[TARGET])})
    result = pd.DataFrame(rows).set_index("col")
    result["p_Holm"] = holm_correction(result["p"])
    with pd.option_context("display.float_format", "{:.4g}".format):
        print(result.to_string())
    print("(V de Cramér ~0.1 pequeño, ~0.3 medio, ~0.5 grande para gl = 1)")
    print("\nTabla A9 x A16 (el predictor individual más fuerte):")
    print(pd.crosstab(df["A9"], df[TARGET], margins=True).to_string())
    return result


def analyse_correlations(df):
    # === 11. Correlaciones entre continuas y colinealidad ===
    section("11. Correlaciones entre continuas y colinealidad")
    # Preferimos Spearman porque las variables no son normales y tienen atípicos; mostramos Pearson para comparar.
    pearson = df[CONTINUOUS].corr(method="pearson")
    spearman = df[CONTINUOUS].corr(method="spearman")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print("Pearson:\n" + pearson.to_string())
        print("Spearman:\n" + spearman.to_string())
    upper = spearman.where(np.triu(np.ones(spearman.shape, dtype=bool), k=1)).stack()
    print(f"Máximo |Spearman| entre pares: {upper.abs().max():.3f} ({upper.abs().idxmax()})")

    # Calculamos el VIF sobre log1p para detectar colinealidad multivariante, no solo por pares.
    complete = np.log1p(df[CONTINUOUS].dropna())
    vif = {}
    for col in CONTINUOUS:
        y = complete[col].values
        X = np.column_stack([np.ones(len(complete)), complete.drop(columns=col).values])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        vif[col] = 1 / (1 - r2)
    print("VIF (log1p, filas completas):", {k: round(float(v), 2) for k, v in vif.items()}, "(> 5 sería preocupante)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, matrix, name in [(axes[0], pearson, "Pearson"), (axes[1], spearman, "Spearman")]:
        im = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(CONTINUOUS)), CONTINUOUS)
        ax.set_yticks(range(len(CONTINUOUS)), CONTINUOUS)
        for i in range(len(CONTINUOUS)):
            for j in range(len(CONTINUOUS)):
                value = matrix.values[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(value) > 0.6 else INK)
        ax.set_title(f"Correlación de {name}")
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.85, label="Coeficiente")
    fig.suptitle("Correlaciones entre variables continuas", fontweight="bold")
    save_figure(fig, "correlation_continuous", "Pearson and Spearman correlation heatmaps of continuous columns")
    return spearman, vif


def analyse_associations(df):
    # === 12. Asociaciones entre todas las variables ===
    section("12. Asociaciones entre todas las variables")
    # Combinamos medidas en [0, 1]: V de Cramér (cat-cat), eta sobre rangos (cat-cont) y |Spearman| (cont-cont).
    cols = [c for c in COLUMNS]
    matrix = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    categorical_like = CATEGORICAL + [TARGET]
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if a in categorical_like and b in categorical_like:
                value = cramers_v(df[a], df[b])
            elif a in categorical_like:
                value = correlation_ratio(df[a], df[b])
            elif b in categorical_like:
                value = correlation_ratio(df[b], df[a])
            else:
                value = abs(df[[a, b]].corr(method="spearman").iloc[0, 1])
            matrix.loc[a, b] = matrix.loc[b, a] = value
    upper = matrix.where(np.triu(np.ones(matrix.shape, dtype=bool), k=1)).stack().sort_values(ascending=False)
    print("Pares más asociados:")
    print(upper.head(10).round(3).to_string())

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(matrix.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45)
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            value = matrix.values[i, j]
            if i != j:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if value > 0.55 else INK)
    ax.set_title("Asociación entre variables\n(V de Cramér cat-cat, eta sobre rangos cat-cont, |Spearman| cont-cont)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Fuerza de asociación [0, 1]")
    save_figure(fig, "association_matrix", "mixed-type association matrix of all columns")
    return matrix


def analyse_scatter_matrix(df):
    # === 13. Matriz de dispersión de continuas por clase ===
    section("13. Matriz de dispersión de continuas por clase")
    # Usamos log1p y las filas completas para ver si alguna pareja separa las clases visualmente.
    data = np.log1p(df[CONTINUOUS + [TARGET]].dropna().set_index(TARGET)).reset_index()
    n = len(CONTINUOUS)
    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    for i, row_col in enumerate(CONTINUOUS):
        for j, col_col in enumerate(CONTINUOUS):
            ax = axes[i, j]
            if i == j:
                for cls in ["-", "+"]:
                    ax.hist(data.loc[data[TARGET] == cls, row_col], bins=25, alpha=0.55,
                            color=CLASS_COLORS[cls], density=True)
                # Ocultamos el eje y de la diagonal porque es densidad, no la escala de la fila.
                ax.set_yticks([])
            else:
                for cls in ["-", "+"]:
                    sub = data[data[TARGET] == cls]
                    ax.scatter(sub[col_col], sub[row_col], s=5, alpha=0.5, color=CLASS_COLORS[cls], linewidths=0)
            if i == n - 1:
                ax.set_xlabel(f"log1p({col_col})")
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(f"log1p({row_col})")
            else:
                ax.set_yticklabels([])
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=CLASS_COLORS[c], label=CLASS_LABELS[c]) for c in ["+", "-"]]
    fig.legend(handles=handles, loc="upper right")
    fig.suptitle(f"Matriz de dispersión de continuas en log1p (n = {len(data)} filas completas)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.95, 0.97))
    save_figure(fig, "scatter_matrix", "scatter matrix of log1p continuous columns by target")
    print(f"Filas usadas: {len(data)}")


def analyse_mutual_information(df):
    # === 14. Información mutua con el objetivo ===
    section("14. Información mutua con el objetivo")
    # La información mutua capta dependencias no lineales y da una referencia de importancia para comparar con XAI.
    complete = df.dropna().copy()
    X = complete.drop(columns=TARGET)
    discrete = []
    for col in X.columns:
        if col in CATEGORICAL:
            X[col] = pd.Categorical(X[col], categories=DOCUMENTED_LEVELS[col]).codes
            discrete.append(True)
        else:
            discrete.append(False)
    y = (complete[TARGET] == "+").astype(int)
    mi = pd.Series(mutual_info_classif(X, y, discrete_features=discrete, random_state=SEED),
                   index=X.columns).sort_values(ascending=False)
    print(f"Filas completas: {len(complete)}")
    print(mi.round(4).to_string())

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.barh(mi.index[::-1], mi.values[::-1], color=COLOR_POS, height=0.6)
    for k, v in enumerate(mi.values[::-1]):
        ax.text(v + 0.003, k, f"{v:.3f}", va="center", fontsize=7.5, color=INK)
    ax.set_xlabel("Información mutua con A16 (nats)")
    ax.set_title("Información mutua de cada atributo con el objetivo")
    ax.grid(axis="y", visible=False)
    save_figure(fig, "mutual_information", "mutual information of each attribute with the target")
    return mi


def run_analysis():
    raw, df = load_for_analysis()
    analyse_schema(raw, df)
    analyse_missing(df)
    analyse_validity(df)
    analyse_duplicates_cardinality(df)
    analyse_continuous(df)
    analyse_categorical(df)
    analyse_target(df)
    analyse_continuous_vs_target(df)
    analyse_categorical_vs_target(df)
    analyse_correlations(df)
    analyse_associations(df)
    analyse_scatter_matrix(df)
    analyse_mutual_information(df)


if __name__ == "__main__":
    run_analysis()
    # PARTE 2: LIMPIEZA, importada de cleanup.py (autocontenida: no usa ningún estado del análisis).
    run_cleanup()
