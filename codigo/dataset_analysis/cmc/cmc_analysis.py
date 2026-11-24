"""Análisis exploratorio y limpieza del dataset Contraceptive Method Choice (UCI, cmc).

Ejecución: python cmc_analysis.py (desde cualquier carpeta); genera img/ y, con cleanup.py, cmc_cleaned.csv.
"""

# === 0. Configuración ===
import hashlib
import importlib.util
import sys
from pathlib import Path

import matplotlib

# Usamos el backend Agg para poder ejecutar el script sin entorno gráfico.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

# La limpieza vive solo en cleanup.py: la cargamos por ruta (sin tocar sys.path) y sin escribir __pycache__.
# La registramos en sys.modules para que sus objetos se puedan serializar con pickle dentro de esta sesión.
_write_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
_spec = importlib.util.spec_from_file_location("cmc_cleanup", Path(__file__).resolve().parent / "cleanup.py")
cmc_cleanup = sys.modules["cmc_cleanup"] = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(cmc_cleanup)
except BaseException:
    del sys.modules["cmc_cleanup"]
    raise
finally:
    sys.dont_write_bytecode = _write_bytecode
run_cleanup, section = cmc_cleanup.run_cleanup, cmc_cleanup.section

SEED = 42
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_PATH = SCRIPT_DIR / ".." / ".." / "datasets" / "cmc" / "cmc.csv"
IMG_DIR = SCRIPT_DIR / "img"
DPI = 200
# Huella del cmc.csv que comprobamos valor a valor contra UCI (ucimlrepo id 30).
RAW_SHA256 = "ad2a49da55df24b061a994ecc4d90c856aba35acd05138851aa251c4f06303c9"

COLUMNS = ["wife_age", "wife_edu", "husband_edu", "num_children", "wife_religion", "wife_working",
           "husband_occupation", "standard_of_living_index", "media_exposure", "contraceptive_method"]
TARGET = "contraceptive_method"
FEATURES = COLUMNS[:-1]

# Tipo de cada atributo según cmc.names; las "categóricas" con 1=low..4=high son ordinales.
NUMERIC = ["wife_age", "num_children"]
ORDINAL = ["wife_edu", "husband_edu", "standard_of_living_index"]
BINARY = ["wife_religion", "wife_working", "media_exposure"]
NOMINAL = ["husband_occupation"]
CATEGORICAL = ORDINAL + BINARY + NOMINAL
DOCUMENTED_CODES = {
    "wife_edu": [1, 2, 3, 4],
    "husband_edu": [1, 2, 3, 4],
    "wife_religion": [0, 1],
    "wife_working": [0, 1],
    "husband_occupation": [1, 2, 3, 4],
    "standard_of_living_index": [1, 2, 3, 4],
    "media_exposure": [0, 1],
    TARGET: [1, 2, 3],
}
DOCUMENTED_MEANING = {
    "wife_religion": "0=No islam, 1=Islam",
    "wife_working": "0=Sí trabaja, 1=No trabaja",
    "media_exposure": "0=Buena, 1=No buena",
}
# La encuesta (NICPS 1987) recoge mujeres casadas en edad fértil, de 15 a 49 años.
AGE_RANGE = (15, 49)
CLASSES = [1, 2, 3]
CLASS_LABELS = {1: "Sin método (1)", 2: "Larga duración (2)", 3: "Corta duración (3)"}
CLASS_SHORT = {1: "Ninguno", 2: "Largo", 3: "Corto"}
# Comprobamos que el esquema del análisis coincide con el de cleanup.py, para que no se separen.
assert COLUMNS == cmc_cleanup.CLEAN_COLUMNS and AGE_RANGE == cmc_cleanup.CLEAN_AGE_RANGE and SEED == cmc_cleanup.SEED
assert {c: v for c, v in DOCUMENTED_CODES.items() if c != TARGET} == cmc_cleanup.CLEAN_CODES
assert (TARGET, NUMERIC, ORDINAL, BINARY) == (cmc_cleanup.CLEAN_TARGET, cmc_cleanup.CLEAN_NUMERIC,
                                             cmc_cleanup.CLEAN_ORDINAL, cmc_cleanup.CLEAN_BINARY)
assert NOMINAL == list(cmc_cleanup.CLEAN_NOMINAL)

# Colores fijos por clase en todas las figuras.
CLASS_COLORS = {1: "#159f7c", 2: "#2a78d6", 3: "#eb6834"}
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
    """Razón de correlación eta (sobre rangos) entre una categórica y una numérica u ordinal."""
    # Usamos rangos para que eta sea robusta a las colas y válida con ordinales.
    vals = values.astype(float).rank()
    grand_mean = vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for _, g in vals.groupby(categories))
    ss_total = ((vals - grand_mean) ** 2).sum()
    return np.sqrt(ss_between / ss_total) if ss_total > 0 else np.nan


def kruskal_epsilon(values, groups):
    """Kruskal-Wallis y su tamaño de efecto épsilon² = H / (n - 1) (Tomczak y Tomczak, 2014)."""
    samples = [values[groups == g] for g in CLASSES]
    h, p = stats.kruskal(*samples)
    return h, p, h / (len(values) - 1)


def dunn_test(values, groups):
    """Post-hoc de Dunn (rangos conjuntos, corrección por empates) con Holm sobre los 3 pares."""
    ranks = values.rank()
    n = len(values)
    tie_counts = values.value_counts().values
    tie_term = (tie_counts ** 3 - tie_counts).sum() / (12 * (n - 1))
    rows = []
    for a, b in [(1, 2), (1, 3), (2, 3)]:
        ra, rb = ranks[groups == a], ranks[groups == b]
        se = np.sqrt((n * (n + 1) / 12 - tie_term) * (1 / len(ra) + 1 / len(rb)))
        z = (ra.mean() - rb.mean()) / se
        rows.append({"par": f"{a}-{b}", "z": z, "p": 2 * stats.norm.sf(abs(z))})
    result = pd.DataFrame(rows).set_index("par")
    result["p_Holm"] = holm_correction(result["p"])
    return result


def adjusted_residuals(table):
    """Residuos estandarizados ajustados de Haberman: ~N(0, 1) en cada celda bajo independencia."""
    observed = table.values.astype(float)
    n = observed.sum()
    rows = observed.sum(axis=1, keepdims=True)
    cols = observed.sum(axis=0, keepdims=True)
    expected = rows @ cols / n
    residuals = (observed - expected) / np.sqrt(expected * (1 - rows / n) * (1 - cols / n))
    return pd.DataFrame(residuals, index=table.index, columns=table.columns)


# ============================================================================
# PARTE 1: ANÁLISIS
# ============================================================================

def load_for_analysis():
    # === 1. Carga y procedencia ===
    section("1. Carga y procedencia")
    # Leemos sin cabecera porque cmc.csv es una copia de cmc.data de UCI, que no la tiene.
    digest = hashlib.sha256(RAW_PATH.read_bytes()).hexdigest()
    print(f"Fichero: {RAW_PATH.resolve().name} | SHA-256 {digest}")
    print(f"Coincide con la copia verificada contra UCI (ucimlrepo id 30): {digest == RAW_SHA256}")
    raw = pd.read_csv(RAW_PATH, header=None, names=COLUMNS, dtype=str, keep_default_na=False)
    print(f"Forma cruda: {raw.shape} (documentado en cmc.names: 1473 x 10)")
    assert raw.shape == (1473, 10), "El fichero no coincide con la documentación de UCI"
    # Exigimos que todo sea entero porque cmc.names solo define códigos enteros y no documenta ausentes.
    non_integer = int((~raw.apply(lambda s: s.str.fullmatch(r"-?\d+"))).sum().sum())
    print(f"Celdas no enteras (vacías, '?', texto o decimales): {non_integer}")
    df = raw.astype(int)
    print("\nPrimeras filas:")
    print(df.head().to_string())
    return df


def analyse_schema(df):
    # === 2. Esquema y tipos ===
    section("2. Esquema y tipos")
    # Clasificamos cada atributo según cmc.names; el tipo decide test, codificación y figura.
    kinds = {**{c: "numérica" for c in NUMERIC}, **{c: "ordinal (1=bajo..4=alto)" for c in ORDINAL},
             **{c: "binaria" for c in BINARY}, **{c: "nominal (sin orden documentado)" for c in NOMINAL},
             TARGET: "objetivo nominal (3 clases)"}
    rows = [{"columna": c, "tipo": kinds[c], "dtype": str(df[c].dtype), "n_unicos": df[c].nunique(),
             "min": df[c].min(), "max": df[c].max()} for c in COLUMNS]
    print(pd.DataFrame(rows).set_index("columna").to_string())
    print("\nSignificado de las binarias (cmc.names):", DOCUMENTED_MEANING)
    print("Ojo: en wife_working y media_exposure el 1 es la respuesta negativa ('No', 'No buena').")


def analyse_missing(df):
    # === 3. Valores ausentes ===
    section("3. Valores ausentes")
    # cmc.names declara 'None'; aun así buscamos NaN y centinelas (0 en códigos 1..4, 99, negativos).
    print(f"NaN totales: {int(df.isna().sum().sum())}")
    sentinels = {c: int(((df[c] == 0) & (min(DOCUMENTED_CODES.get(c, [0])) == 1)).sum()
                        + (df[c] >= 99).sum() + (df[c] < 0).sum()) for c in COLUMNS}
    print(f"Posibles centinelas por columna: {sentinels}")
    print("Conclusión: no hay ausentes, así que la imputación por partición solo actúa como red de seguridad.")


def analyse_validity(df):
    # === 4. Códigos inválidos y consistencia ===
    section("4. Códigos inválidos y consistencia")
    # Comprobamos cada código contra cmc.names; uno fuera del esquema sería un error de transcripción.
    for col, codes in DOCUMENTED_CODES.items():
        invalid = int((~df[col].isin(codes)).sum())
        unused = sorted(set(codes) - set(df[col].unique()))
        print(f"  {col}: fuera del esquema={invalid}; códigos documentados sin uso={unused or 'ninguno'}")
    low, high = AGE_RANGE
    print(f"\nwife_age fuera de [{low}, {high}]: {int((~df['wife_age'].between(low, high)).sum())}"
          f" (observado {df['wife_age'].min()}-{df['wife_age'].max()})")
    print(f"num_children negativos: {int((df['num_children'] < 0).sum())}; máximo: {df['num_children'].max()}")
    # Contrastamos hijos frente a edad: más hijos que años desde los 13 sería biológicamente dudoso.
    years_fertile = (df["wife_age"] - 13).clip(lower=1)
    implausible = df[df["num_children"] > years_fertile]
    print(f"Filas con num_children > edad - 13: {len(implausible)}")
    if len(implausible):
        print(implausible[["wife_age", "num_children"]].to_string())
    print("Máximo de hijos por edad (16-25):", df.groupby("wife_age")["num_children"].max().loc[16:25].to_dict())
    # Coherencia lógica: sin hijos casi nadie usa un método de larga duración (esterilización, DIU).
    print("\nnum_children == 0 frente al objetivo:", df.loc[df["num_children"] == 0, TARGET].value_counts()
          .reindex(CLASSES, fill_value=0).to_dict())


def analyse_duplicates_cardinality(df):
    # === 5. Duplicados, columnas casi constantes y cardinalidad ===
    section("5. Duplicados, columnas casi constantes y cardinalidad")
    # Separamos duplicados exactos y de atributos: estos últimos revelan perfiles con etiquetas distintas.
    exact = df.duplicated()
    in_groups = df.duplicated(keep=False)
    print(f"Filas duplicadas exactas: {int(exact.sum())} ({100 * exact.mean():.1f} %), "
          f"en {df[in_groups].groupby(COLUMNS).ngroups} grupos que suman {int(in_groups.sum())} filas")
    feature_dups = df.duplicated(subset=FEATURES)
    profiles = df.groupby(FEATURES)[TARGET].nunique()
    print(f"Filas duplicadas en atributos (ignorando el objetivo): {int(feature_dups.sum())}")
    print(f"Perfiles distintos: {len(profiles)}; perfiles con etiquetas contradictorias: {int((profiles > 1).sum())}")
    # Si fueran copias de transcripción aparecerían contiguas; medimos la distancia en el orden del fichero.
    gaps = np.concatenate([np.diff(g.index.values) for _, g in df[in_groups].groupby(COLUMNS)])
    print(f"Distancia entre copias en el fichero: mínima {gaps.min()}, mediana {np.median(gaps):.0f}, "
          f"contiguas {int((gaps == 1).sum())}")
    print("Perfil medio de las filas duplicadas frente al global:")
    print(pd.DataFrame({"duplicadas": df[exact].mean(), "global": df.mean()}).round(2).T.to_string())
    print(f"Clases de las copias: {df.loc[exact, TARGET].value_counts().reindex(CLASSES).to_dict()}")
    # Contrastamos copias frente a mujeres distintas: una copia repite la etiqueta siempre; dos mujeres
    # con el mismo perfil coinciden con probabilidad sum_k p_k(x)^2, que estimamos fuera de pliegue.
    groups = df.groupby(FEATURES).ngroup()
    model = make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=2000))
    proba = cross_val_predict(model, df[FEATURES], df[TARGET], groups=groups, cv=GroupKFold(n_splits=5),
                              method="predict_proba")
    agreement = pd.Series((proba ** 2).sum(axis=1), index=df.index)
    per_group = []
    for _, g in df[df.duplicated(subset=FEATURES, keep=False)].groupby(FEATURES):
        labels = g[TARGET].values
        n_pairs = len(g) * (len(g) - 1) / 2
        same = sum(labels[i] == labels[j] for i in range(len(g)) for j in range(i + 1, len(g)))
        per_group.append((n_pairs, same, n_pairs * agreement[g.index].mean()))
    per_group = np.array(per_group, dtype=float)

    def copy_fraction(rows):
        obs, exp = rows[:, 1].sum() / rows[:, 0].sum(), rows[:, 2].sum() / rows[:, 0].sum()
        return obs, exp, (obs - exp) / (1 - exp)

    obs, exp, frac = copy_fraction(per_group)
    # Remuestreamos perfiles (bootstrap por grupos, semilla fija) para dar un intervalo a la fracción de copias.
    rng = np.random.default_rng(SEED)
    boot = [copy_fraction(per_group[rng.integers(0, len(per_group), len(per_group))])[2] for _ in range(2000)]
    low, high = np.percentile(boot, [2.5, 97.5])
    print(f"Pares con el mismo perfil: {int(per_group[:, 0].sum())} en {len(per_group)} perfiles; misma etiqueta "
          f"observada {obs:.3f} frente a {exp:.3f} esperada si son mujeres distintas")
    print(f"Fracción de copias implícita: {frac:.3f} (IC bootstrap 95 % [{low:.3f}, {high:.3f}])")

    # Medimos la frecuencia del valor dominante; por encima del 95 % la columna aporta poca variación.
    rows = []
    for col in COLUMNS:
        counts = df[col].value_counts(normalize=True)
        rows.append({"columna": col, "n_unicos": df[col].nunique(), "valor_dominante": counts.index[0],
                     "%_dominante": round(100 * counts.iloc[0], 1)})
    cardinality = pd.DataFrame(rows).set_index("columna")
    print(cardinality.to_string())
    near_constant = cardinality[cardinality["%_dominante"] > 95].index.tolist()
    print(f"Columnas casi constantes (>95 %): {near_constant or 'ninguna'}")


def analyse_numeric(df):
    # === 6. Estadística univariante: numéricas ===
    section("6. Estadística univariante: numéricas")
    rows = []
    for col in NUMERIC:
        x = df[col]
        q1, q3 = x.quantile([0.25, 0.75])
        rows.append({"col": col, "media": x.mean(), "mediana": x.median(), "std": x.std(), "min": x.min(),
                     "Q1": q1, "Q3": q3, "max": x.max(), "asimetría": stats.skew(x),
                     "curtosis_exc": stats.kurtosis(x), "asim_sqrt": stats.skew(np.sqrt(x)),
                     "asim_log1p": stats.skew(np.log1p(x))})
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(pd.DataFrame(rows).set_index("col").to_string())
    # Contrastamos normalidad con Shapiro-Wilk y D'Agostino K² para decidir entre tests paramétricos o de rangos.
    print("\nNormalidad:")
    for col in NUMERIC:
        x = df[col]
        print(f"  {col}: Shapiro W={stats.shapiro(x).statistic:.3f} p={stats.shapiro(x).pvalue:.2e} | "
              f"D'Agostino K²={stats.normaltest(x).statistic:.1f} p={stats.normaltest(x).pvalue:.2e}")
    # Usamos la regla IQR (robusta) y |z| > 3 solo como referencia, porque num_children es asimétrica.
    print("\nAtípicos por regla IQR (1.5·IQR) y |z| > 3:")
    for col in NUMERIC:
        x = df[col]
        q1, q3 = x.quantile([0.25, 0.75])
        upper = q3 + 1.5 * (q3 - q1)
        iqr_out = x[(x < q1 - 1.5 * (q3 - q1)) | (x > upper)]
        print(f"  {col}: IQR={len(iqr_out)} ({100 * len(iqr_out) / len(x):.1f} %, umbral superior {upper:.1f}), "
              f"|z|>3={int((np.abs(stats.zscore(x)) > 3).sum())}, valores atípicos={sorted(int(v) for v in iqr_out.unique())}")
    print("Cola de num_children:", df["num_children"].value_counts().sort_index().loc[9:].to_dict())

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, col, label in [(axes[0], "wife_age", "Edad de la mujer (años)"),
                           (axes[1], "num_children", "Nº de hijos nacidos")]:
        x = df[col]
        bins = np.arange(x.min() - 0.5, x.max() + 1.5, 1)
        ax.hist(x, bins=bins, color=CLASS_COLORS[2], alpha=0.6, edgecolor="white", linewidth=0.4)
        ax.axvline(x.median(), color=CLASS_COLORS[3], linestyle="--", linewidth=1.5, label=f"Mediana = {x.median():.0f}")
        ax.set_title(f"{col} (asimetría {stats.skew(x):.2f})")
        ax.set_xlabel(label)
        ax.set_ylabel("Nº de mujeres")
        ax.legend()
    fig.suptitle("Distribución de las variables numéricas", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "numeric_histograms", "histograms of wife_age and num_children")


def analyse_categorical(df):
    # === 7. Estadística univariante: ordinales, binarias y nominal ===
    section("7. Estadística univariante: ordinales, binarias y nominal")
    for col in CATEGORICAL:
        counts = df[col].value_counts().sort_index()
        pct = (100 * counts / len(df)).round(1)
        print(f"  {col}: " + ", ".join(f"{k}={v} ({p} %)" for k, v, p in zip(counts.index, counts, pct)))
    # Señalamos niveles raros (<5 %) porque generan columnas one-hot casi vacías.
    rare = {c: df[c].value_counts()[lambda s: s < 0.05 * len(df)].to_dict() for c in CATEGORICAL}
    print(f"Niveles con <5 % de filas: { {c: v for c, v in rare.items() if v} }")

    # Figura: barras apiladas al 100 % por nivel, porque con 3 clases es lo que mejor muestra la tendencia.
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.8))
    for ax, col in zip(axes.flat, CATEGORICAL):
        table = pd.crosstab(df[col], df[TARGET]).reindex(columns=CLASSES, fill_value=0)
        shares = table.div(table.sum(axis=1), axis=0)
        bottom = np.zeros(len(shares))
        positions = np.arange(len(shares))
        for cls in CLASSES:
            ax.bar(positions, shares[cls], bottom=bottom, color=CLASS_COLORS[cls], width=0.7,
                   label=CLASS_LABELS[cls], edgecolor="white", linewidth=0.6)
            bottom += shares[cls].values
        ax.set_xticks(positions, [f"{lvl}\nn={n}" for lvl, n in zip(table.index, table.sum(axis=1))], fontsize=7.5)
        ax.set_ylim(0, 1)
        ax.set_title(col, fontsize=9)
        ax.set_ylabel("Proporción")
        ax.grid(axis="x", visible=False)
    ax = axes.flat[-1]
    ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[c]) for c in CLASSES]
    ax.legend(handles, [CLASS_LABELS[c] for c in CLASSES], loc="center", title="Método anticonceptivo")
    fig.suptitle("Método anticonceptivo por nivel de cada atributo categórico (barras al 100 %)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "categorical_by_target", "stacked target proportions per level of each categorical attribute")


def analyse_target(df):
    # === 8. Balance del objetivo ===
    section("8. Balance del objetivo")
    counts = df[TARGET].value_counts().reindex(CLASSES)
    print(f"Clases: {counts.to_dict()} | proporciones: {(100 * counts / len(df)).round(1).to_dict()}")
    print(f"Ratio mayoritaria/minoritaria = {counts.max() / counts.min():.2f}; "
          f"tasa de acierto de la clase mayoritaria = {counts.max() / len(df):.3f}")
    # Con un ratio < 2 no remuestreamos, pero estratificamos y medimos con F1 macro además de accuracy.
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bars = ax.bar([CLASS_LABELS[c] for c in CLASSES], counts.values,
                  color=[CLASS_COLORS[c] for c in CLASSES], width=0.55)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 8, f"{value} ({100 * value / len(df):.1f} %)",
                ha="center", va="bottom", color=INK)
    ax.set_ylabel("Nº de mujeres")
    ax.set_title("Balance de la variable objetivo contraceptive_method")
    ax.set_ylim(0, counts.max() * 1.15)
    save_figure(fig, "target_balance", "target class balance")


def analyse_numeric_vs_target(df):
    # === 9. Bivariante: numéricas frente al objetivo ===
    section("9. Bivariante: numéricas frente al objetivo")
    # Usamos Kruskal-Wallis (no ANOVA) porque no hay normalidad y el objetivo tiene 3 grupos independientes.
    y = df[TARGET]
    rows = []
    for col in NUMERIC:
        h, p, eps2 = kruskal_epsilon(df[col], y)
        medians = df.groupby(TARGET)[col].median().to_dict()
        rows.append({"col": col, **{f"mediana_{c}": medians[c] for c in CLASSES}, "H": h, "p": p, "épsilon²": eps2})
    result = pd.DataFrame(rows).set_index("col")
    result["p_Holm"] = holm_correction(result["p"])
    with pd.option_context("display.float_format", "{:.4g}".format):
        print(result.to_string())
    print("(épsilon² ~0.01 pequeño, ~0.08 medio, ~0.26 grande)")
    # Dunn con Holm localiza qué pares de clases difieren, que es lo que Kruskal-Wallis no dice.
    for col in NUMERIC:
        print(f"\nDunn ({col}):")
        with pd.option_context("display.float_format", "{:.3g}".format):
            print(dunn_test(df[col], y).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col in zip(axes, NUMERIC):
        data = [df.loc[y == c, col] for c in CLASSES]
        parts = ax.violinplot(data, showextrema=False, widths=0.8)
        for body, cls in zip(parts["bodies"], CLASSES):
            body.set_facecolor(CLASS_COLORS[cls])
            body.set_alpha(0.35)
        box = ax.boxplot(data, widths=0.25, patch_artist=True, medianprops={"color": INK, "linewidth": 1.5},
                         flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.5, "markeredgecolor": INK_SECONDARY})
        for patch, cls in zip(box["boxes"], CLASSES):
            patch.set_facecolor(CLASS_COLORS[cls])
            patch.set_alpha(0.85)
        ax.set_xticks([1, 2, 3], [CLASS_LABELS[c] for c in CLASSES])
        ax.set_ylabel(col)
        row = result.loc[col]
        ax.set_title(f"{col}: ε² = {row['épsilon²']:.3f}, p Holm = {row['p_Holm']:.1e}")
    fig.suptitle("Variables numéricas por método anticonceptivo (Kruskal-Wallis)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "numeric_by_target", "violin and box plots of numeric columns by target")

    # Figura: proporción de cada método por edad y por hijos, porque la relación con la edad no es monótona.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    age_bins = pd.cut(df["wife_age"], bins=[15, 20, 25, 30, 35, 40, 45, 49], include_lowest=True,
                      labels=["15-20", "21-25", "26-30", "31-35", "36-40", "41-45", "46-49"])
    child_bins = df["num_children"].clip(upper=8)
    for ax, groups, xlabel in [(axes[0], age_bins, "Edad de la mujer (años)"),
                               (axes[1], child_bins, "Nº de hijos (8 = 8 o más)")]:
        table = pd.crosstab(groups, y).reindex(columns=CLASSES, fill_value=0)
        shares = table.div(table.sum(axis=1), axis=0)
        labels = [str(i) for i in table.index]
        for cls in CLASSES:
            ax.plot(range(len(shares)), shares[cls], marker="o", color=CLASS_COLORS[cls], label=CLASS_LABELS[cls])
        ax.set_xticks(range(len(shares)), [f"{l}\nn={n}" for l, n in zip(labels, table.sum(axis=1))], fontsize=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Proporción dentro del grupo")
        ax.set_ylim(0, 1)
    axes[0].legend(loc="upper left")
    fig.suptitle("Método anticonceptivo según edad y número de hijos", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "method_by_age_children", "target proportions by age band and number of children")
    print("\nMétodo por tramo de edad (%):")
    print((100 * pd.crosstab(age_bins, y, normalize="index")).round(1).to_string())
    return result


def analyse_categorical_vs_target(df):
    # === 10. Bivariante: categóricas frente al objetivo ===
    section("10. Bivariante: categóricas frente al objetivo")
    # Chi-cuadrado de independencia (k x 3) con V de Cramér; para ordinales añadimos Kruskal-Wallis, que usa el orden.
    y = df[TARGET]
    rows = []
    for col in CATEGORICAL:
        table = pd.crosstab(df[col], y)
        chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
        row = {"col": col, "niveles": table.shape[0], "chi2": chi2, "gl": dof,
               "min_esperada": expected.min(), "p_chi2": p, "V_Cramer": cramers_v(df[col], y)}
        if col in ORDINAL:
            _, p_kw, eps2 = kruskal_epsilon(df[col], y)
            row.update({"p_KW": p_kw, "épsilon²": eps2})
        rows.append(row)
    result = pd.DataFrame(rows).set_index("col")
    result["p_chi2_Holm"] = holm_correction(result["p_chi2"])
    with pd.option_context("display.float_format", "{:.4g}".format):
        print(result.to_string())
    print("(todas las frecuencias esperadas >= 5, así que la aproximación chi-cuadrado es válida;"
          "\n V de Cramér: con 4 niveles (gl* = 2) ~0.07 pequeño, ~0.21 medio, ~0.35 grande;"
          "\n con las binarias (gl* = 1) ~0.10 pequeño, ~0.30 medio, ~0.50 grande)")

    # Post-hoc: residuos ajustados con Holm dentro de cada tabla, para saber qué celdas causan la asociación.
    residual_tables = {}
    print("\nCeldas significativas (|residuo ajustado|, Holm < 0.05):")
    for col in CATEGORICAL:
        residuals = adjusted_residuals(pd.crosstab(df[col], y))
        p = 2 * stats.norm.sf(np.abs(residuals.values)).ravel()
        p_adj = holm_correction(p).reshape(residuals.shape)
        residual_tables[col] = (residuals, p_adj)
        hits = [f"{col}={lvl}/{CLASS_SHORT[cls]}: {residuals.loc[lvl, cls]:+.1f}"
                for i, lvl in enumerate(residuals.index) for j, cls in enumerate(CLASSES) if p_adj[i, j] < 0.05]
        print(f"  {col}: " + ("; ".join(hits) if hits else "ninguna"))

    fig, axes = plt.subplots(2, 4, figsize=(13, 7), gridspec_kw={"hspace": 0.55})
    for ax, col in zip(axes.flat, CATEGORICAL):
        residuals, p_adj = residual_tables[col]
        im = ax.imshow(residuals.values, cmap="RdBu_r", vmin=-8, vmax=8, aspect="auto")
        for i in range(residuals.shape[0]):
            for j in range(residuals.shape[1]):
                mark = "*" if p_adj[i, j] < 0.05 else ""
                ax.text(j, i, f"{residuals.values[i, j]:+.1f}{mark}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(residuals.values[i, j]) > 5 else INK)
        ax.set_xticks(range(3), [CLASS_SHORT[c] for c in CLASSES])
        ax.set_yticks(range(len(residuals)), residuals.index)
        ax.set_title(f"{col}\nV = {result.loc[col, 'V_Cramer']:.2f}", fontsize=8.5)
        ax.grid(False)
    axes.flat[-1].axis("off")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Residuo estandarizado ajustado")
    fig.suptitle("Residuos ajustados nivel x método (* = Holm < 0.05; rojo = más casos de lo esperado)",
                 fontweight="bold")
    save_figure(fig, "adjusted_residuals", "adjusted standardized residuals of each categorical attribute vs target")
    return result


def analyse_correlations(df):
    # === 11. Correlaciones entre numéricas y ordinales, y colinealidad ===
    section("11. Correlaciones entre numéricas y ordinales, y colinealidad")
    # Preferimos Spearman porque las ordinales solo tienen orden y las numéricas no son normales.
    ordered = NUMERIC + ORDINAL
    spearman = df[ordered].corr(method="spearman")
    kendall = df[ordered].corr(method="kendall")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print("Spearman:\n" + spearman.to_string())
        print("Kendall tau-b (referencia con empates):\n" + kendall.to_string())
    upper = spearman.where(np.triu(np.ones(spearman.shape, dtype=bool), k=1)).stack()
    print(f"Máximo |Spearman| entre pares: {upper.abs().max():.3f} ({upper.abs().idxmax()})")
    # El VIF sobre todos los atributos (nominal en one-hot) detecta colinealidad multivariante.
    X = pd.get_dummies(df[FEATURES], columns=NOMINAL, drop_first=True, dtype=float)
    vif = {}
    for col in X.columns:
        target = X[col].values
        design = np.column_stack([np.ones(len(X)), X.drop(columns=col).values])
        beta, *_ = np.linalg.lstsq(design, target, rcond=None)
        r2 = 1 - ((target - design @ beta) ** 2).sum() / ((target - target.mean()) ** 2).sum()
        vif[col] = 1 / (1 - r2)
    print("VIF:", {k: round(float(v), 2) for k, v in vif.items()}, "(> 5 sería preocupante)")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(spearman.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(ordered)), ordered, rotation=35, ha="right")
    ax.set_yticks(range(len(ordered)), ordered)
    for i in range(len(ordered)):
        for j in range(len(ordered)):
            value = spearman.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > 0.6 else INK)
    ax.set_title("Correlación de Spearman (numéricas y ordinales)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="rho de Spearman")
    save_figure(fig, "correlation_spearman", "Spearman correlation heatmap of numeric and ordinal columns")
    return vif


def analyse_associations(df):
    # === 12. Asociaciones entre todas las variables ===
    section("12. Asociaciones entre todas las variables")
    # Medidas en [0, 1]: |Spearman| entre ordenadas, V de Cramér entre no ordenadas y eta sobre rangos en el resto.
    ordered = set(NUMERIC + ORDINAL)
    unordered = set(BINARY + NOMINAL + [TARGET])
    matrix = pd.DataFrame(np.eye(len(COLUMNS)), index=COLUMNS, columns=COLUMNS)
    for i, a in enumerate(COLUMNS):
        for b in COLUMNS[i + 1:]:
            if a in ordered and b in ordered:
                value = abs(df[[a, b]].corr(method="spearman").iloc[0, 1])
            elif a in unordered and b in unordered:
                value = cramers_v(df[a], df[b])
            else:
                cat, num = (a, b) if a in unordered else (b, a)
                value = correlation_ratio(df[cat], df[num])
            matrix.loc[a, b] = matrix.loc[b, a] = value
    upper = matrix.where(np.triu(np.ones(matrix.shape, dtype=bool), k=1)).stack().sort_values(ascending=False)
    print("Pares más asociados:")
    print(upper.head(10).round(3).to_string())

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(matrix.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(COLUMNS)), COLUMNS, rotation=40, ha="right")
    ax.set_yticks(range(len(COLUMNS)), COLUMNS)
    for i in range(len(COLUMNS)):
        for j in range(len(COLUMNS)):
            if i != j:
                value = matrix.values[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if value > 0.55 else INK)
    ax.set_title("Asociación entre variables\n(|Spearman| ordenadas, V de Cramér no ordenadas, eta sobre rangos mixtas)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Fuerza de asociación [0, 1]")
    save_figure(fig, "association_matrix", "mixed-type association matrix of all columns")


def analyse_age_children(df):
    # === 13. Interacción edad x hijos por clase ===
    section("13. Interacción edad x hijos por clase")
    # Ambas son discretas; añadimos un pequeño ruido solo al dibujar para que los puntos no se solapen.
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
    for ax, cls in zip(axes, CLASSES):
        sub = df[df[TARGET] == cls]
        ax.scatter(sub["wife_age"] + rng.uniform(-0.35, 0.35, len(sub)),
                   sub["num_children"] + rng.uniform(-0.3, 0.3, len(sub)),
                   s=5, alpha=0.45, color=CLASS_COLORS[cls], linewidths=0)
        ax.set_title(f"{CLASS_LABELS[cls]} (n = {len(sub)})")
        ax.set_xlabel("Edad de la mujer (años)")
    axes[0].set_ylabel("Nº de hijos")
    fig.suptitle("Edad frente a número de hijos por método (con ruido de dibujo)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "age_children_by_target", "jittered scatter of wife_age vs num_children per target class")
    # Cuantificamos la pauta: hijos por año de edad fértil, mediana por clase.
    rate = df["num_children"] / (df["wife_age"] - 13).clip(lower=1)
    print("Mediana de hijos por año desde los 13, por clase:", rate.groupby(df[TARGET]).median().round(3).to_dict())


def analyse_mutual_information(df):
    # === 14. Información mutua con el objetivo ===
    section("14. Información mutua con el objetivo")
    # La información mutua capta dependencias no lineales y es una referencia de importancia para comparar con XAI.
    discrete = [c not in NUMERIC for c in FEATURES]
    mi = pd.Series(mutual_info_classif(df[FEATURES], df[TARGET], discrete_features=discrete, random_state=SEED),
                   index=FEATURES).sort_values(ascending=False)
    print(mi.round(4).to_string())
    print(f"Entropía del objetivo: {stats.entropy(df[TARGET].value_counts(normalize=True)):.4f} nats")

    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.barh(mi.index[::-1], mi.values[::-1], color=CLASS_COLORS[2], height=0.6)
    for k, v in enumerate(mi.values[::-1]):
        ax.text(v + 0.001, k, f"{v:.3f}", va="center", fontsize=7.5, color=INK)
    ax.set_xlabel("Información mutua con contraceptive_method (nats)")
    ax.set_title("Información mutua de cada atributo con el objetivo")
    ax.grid(axis="y", visible=False)
    save_figure(fig, "mutual_information", "mutual information of each attribute with the target")


def run_analysis():
    df = load_for_analysis()
    analyse_schema(df)
    analyse_missing(df)
    analyse_validity(df)
    analyse_duplicates_cardinality(df)
    analyse_numeric(df)
    analyse_categorical(df)
    analyse_target(df)
    analyse_numeric_vs_target(df)
    analyse_categorical_vs_target(df)
    analyse_correlations(df)
    analyse_associations(df)
    analyse_age_children(df)
    analyse_mutual_information(df)


if __name__ == "__main__":
    run_analysis()
    # PARTE 2: LIMPIEZA, importada de cleanup.py (autocontenida: no usa ningún estado del análisis).
    run_cleanup()
