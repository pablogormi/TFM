"""Análisis exploratorio del dataset Sick (Thyroid Disease, UCI) seguido de su limpieza.

Ejecución: python sick_analysis.py; une sick.data y sick.test (3772 filas), genera img/ y escribe sick_cleaned.csv.
"""

# === 0. Configuración ===
import importlib.util
import re
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
_spec = importlib.util.spec_from_file_location("sick_cleanup", Path(__file__).resolve().parent / "cleanup.py")
sick_cleanup = importlib.util.module_from_spec(_spec)
# La registramos en sys.modules para que pickle encuentre SickPreprocessor dentro de esta ejecución.
sys.modules["sick_cleanup"] = sick_cleanup
_spec.loader.exec_module(sick_cleanup)
run_cleanup, section = sick_cleanup.run_cleanup, sick_cleanup.section

SEED = 42
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / ".." / ".." / "datasets" / "sick"
RAW_PATHS = [DATA_DIR / "sick.data", DATA_DIR / "sick.test"]
HAND_CLEANED_PATH = DATA_DIR / "sick-cleaned.csv"
DOCUMENTED_ROWS = {"sick.data": 2800, "sick.test": 972}
IMG_DIR = SCRIPT_DIR / "img"
DPI = 200

# Nombres de codigo/main.ipynb (celda 12), extraídos de sick.names.
COLUMNS = ["age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication", "sick",
           "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid", "query_hyperthyroid",
           "lithium", "goitre", "tumor", "hypopituitary", "psych", "TSH_measured", "TSH", "T3_measured",
           "T3", "TT4_measured", "TT4", "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured",
           "TBG", "referral_source", "label"]
TARGET = "label"
# El sufijo `.|<id>` de cada línea de sick.data y sick.test: el punto cierra la clase y el número es el id del registro.
SUFFIX_REGEX = re.compile(r"\.\|(\d+)$")

LABS = ["TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
MEASURED = {lab: f"{lab}_measured" for lab in LABS}
CONTINUOUS = ["age"] + LABS
HISTORY = ["on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication", "sick", "pregnant",
           "thyroid_surgery", "I131_treatment", "query_hypothyroid", "query_hyperthyroid", "lithium",
           "goitre", "tumor", "hypopituitary", "psych"]
BINARY_TF = HISTORY + list(MEASURED.values())
DOCUMENTED_LEVELS = {"sex": ["M", "F"], **{c: ["f", "t"] for c in BINARY_TF},
                     "referral_source": ["WEST", "STMW", "SVHC", "SVI", "SVHD", "other"],
                     TARGET: ["negative", "sick"]}
CATEGORICAL = ["sex"] + BINARY_TF + ["referral_source"]

# Colores: azul para negativo y naranja para sick (la clase minoritaria), fijos en todas las figuras.
COLOR_NEG = "#2a78d6"
COLOR_SICK = "#eb6834"
CLASS_COLORS = {"negative": COLOR_NEG, "sick": COLOR_SICK}
CLASS_LABELS = {"negative": "Negativo", "sick": "Sick"}
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


def haldane_odds_ratio(table):
    """Odds ratio con corrección de Haldane (+0.5) para que no explote con celdas a cero."""
    a, b, c, d = (table.values.astype(float) + 0.5).ravel()
    return (a * d) / (b * c)


# ============================================================================
# PARTE 1: ANÁLISIS
# ============================================================================

def read_sick_data_lines(path):
    """Leemos un fichero de UCI, separamos el sufijo `.|<id>` y devolvemos (líneas sin sufijo, ids)."""
    lines, ids = [], []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        match = SUFFIX_REGEX.search(line)
        assert match, f"Línea {number} sin el sufijo .|<id>"
        lines.append(line[:match.start()])
        ids.append(int(match.group(1)))
    return lines, ids


def load_for_analysis():
    # === 1. Carga y procedencia ===
    section("1. Carga y procedencia")
    # Derivamos los datos de los ficheros de UCI para no depender de la edición manual de sick-cleaned.csv.
    parts = {path.name: read_sick_data_lines(path) for path in RAW_PATHS}
    for name, (lines, ids) in parts.items():
        print(f"{name}: {len(lines)} líneas (documentado {DOCUMENTED_ROWS[name]}), "
              f"campos {sorted({l.count(',') + 1 for l in lines})}, ids en [{min(ids)}, {max(ids)}]")
        assert len(lines) == DOCUMENTED_ROWS[name], f"{name} no coincide con la documentación de UCI"
    # Comparamos línea a línea y valor a valor con la versión de sick.data editada a mano (sick-cleaned.csv).
    data_lines = parts["sick.data"][0]
    hand_lines = HAND_CLEANED_PATH.read_text().splitlines()
    line_diffs = [i for i, (a, b) in enumerate(zip(data_lines, hand_lines)) if a != b]
    data_part = pd.DataFrame([l.split(",") for l in data_lines], columns=COLUMNS)
    hand = pd.read_csv(HAND_CLEANED_PATH, header=None, names=COLUMNS, dtype=str, keep_default_na=False)
    cell_diffs = int((data_part.values != hand.values).sum()) if data_part.shape == hand.shape else -1
    print(f"sick.data frente a sick-cleaned.csv: líneas distintas {len(line_diffs)}, celdas distintas {cell_diffs}, "
          f"forma {data_part.shape} vs {hand.shape}")
    print(f"sick-cleaned.csv == sick.data sin sufijo: {len(line_diffs) == 0 and cell_diffs == 0 and len(data_lines) == len(hand_lines)}")
    # Unimos ambas partes; los ids deben ser únicos y cubrir 1..3772 si son una partición del original.
    lines = [l for name in parts for l in parts[name][0]]
    ids = [i for name in parts for i in parts[name][1]]
    sources = [name for name in parts for _ in parts[name][0]]
    print(f"Ids unidos: {len(set(ids))} únicos de {len(ids)}; cubren 1..{len(ids)}: {sorted(ids) == list(range(1, len(ids) + 1))}")
    derived = pd.DataFrame([l.split(",") for l in lines], columns=COLUMNS)
    assert derived.shape == (3772, 30), "La unión no tiene 3772 x 30"
    question_marks = (derived == "?").sum()
    print(f"Marcadores '?' totales: {int(question_marks.sum())}")
    # Convertimos los `?` a NaN porque es el marcador de ausente de UCI.
    df = derived.mask(derived == "?")
    for col in CONTINUOUS:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df["record_id"] = ids
    df["source"] = sources
    # Comparamos las dos partes de UCI para comprobar que vienen de la misma distribución antes de unirlas.
    print("\nComparación sick.data frente a sick.test:")
    source_table = pd.crosstab(df["source"], df[TARGET])
    print(source_table.to_string())
    print(f"  % sick: {(100 * source_table['sick'] / source_table.sum(axis=1)).round(2).to_dict()}, "
          f"Fisher p={stats.fisher_exact(source_table)[1]:.3f}")
    for col in CONTINUOUS[:-1]:
        a = df.loc[df["source"] == "sick.data", col].dropna()
        b = df.loc[df["source"] == "sick.test", col].dropna()
        ks = stats.ks_2samp(a, b)
        print(f"  {col}: mediana {a.median():.2f} vs {b.median():.2f}, KS D={ks.statistic:.3f} p={ks.pvalue:.3f}")
    missing_by_source = df.groupby("source")[CONTINUOUS + ["sex"]].apply(lambda g: (100 * g.isna().mean()).round(1))
    print("  % ausentes por parte:\n" + missing_by_source.to_string())
    print("\nPrimeras filas:")
    print(df.head().to_string())
    return derived, df


def analyse_schema(raw, df):
    # === 2. Esquema y tipos ===
    section("2. Esquema y tipos")
    rows = []
    for col in COLUMNS:
        kind = "continua" if col in CONTINUOUS else ("objetivo" if col == TARGET else "categórica")
        values = df[col].dropna()
        is_integer = bool(kind == "continua" and len(values) and np.all(np.mod(values, 1) == 0))
        rows.append({"columna": col, "tipo_doc": kind, "dtype": str(df[col].dtype),
                     "n_unicos": values.nunique(), "entera": is_integer if kind == "continua" else "-"})
    print(pd.DataFrame(rows).set_index("columna").to_string())
    # Buscamos niveles no documentados porque indicarían erratas o un esquema distinto.
    print("\nNiveles observados frente a documentados (solo columnas con alguna diferencia):")
    for col, documented in DOCUMENTED_LEVELS.items():
        observed = set(df[col].dropna().unique())
        undocumented = sorted(observed - set(documented))
        unused = sorted(set(documented) - observed)
        if undocumented or unused:
            print(f"  {col}: no documentados={undocumented or 'ninguno'}; documentados sin uso={unused or 'ninguno'}")
    odd = int(sum((raw[c].str.strip() != raw[c]).sum() for c in COLUMNS))
    print(f"Valores con espacios al principio o al final: {odd}")


def analyse_missing(df):
    # === 3. Valores ausentes ===
    section("3. Valores ausentes")
    missing_counts = df[COLUMNS].isna().sum()
    table = pd.DataFrame({"ausentes": missing_counts, "%": (100 * missing_counts / len(df)).round(2)})
    print(table[table.ausentes > 0].to_string())
    row_missing = df[COLUMNS].drop(columns="TBG").isna().sum(axis=1)
    rows_with_missing = int((row_missing > 0).sum())
    print(f"Sin contar TBG: filas con algún ausente {rows_with_missing} ({100 * rows_with_missing / len(df):.1f} %)"
          f" | completas {len(df) - rows_with_missing}")
    print("Nº de ausentes por fila (sin TBG):", row_missing.value_counts().sort_index().to_dict())

    # Comprobamos que cada flag *_measured = f equivale exactamente a que falte su medida.
    print("\nConsistencia flag *_measured frente a ausencia de la medida:")
    for lab, flag in MEASURED.items():
        agree = ((df[flag] == "f") == df[lab].isna()).mean()
        print(f"  {flag}: f={int((df[flag] == 'f').sum())}, NaN en {lab}={int(df[lab].isna().sum())}, acuerdo={100 * agree:.1f} %")

    # Contamos la co-ocurrencia porque un bloque de analíticas ausentes juntas sugiere que no se pidió la prueba.
    indicator = df[COLUMNS].isna().astype(int)
    missing_cols = [c for c in COLUMNS if 0 < missing_counts[c] < len(df)]
    co_occurrence = indicator[missing_cols].T.dot(indicator[missing_cols])
    print("\nCo-ocurrencia de ausentes (sin TBG, ausente siempre):")
    print(co_occurrence.to_string())
    patterns = indicator[missing_cols][row_missing > 0].astype(str).agg("".join, axis=1)
    pattern_names = patterns.map(lambda s: "+".join(c for c, b in zip(missing_cols, s) if b == "1"))
    print("Patrones de ausencia más frecuentes:", pattern_names.value_counts().head(8).to_dict())

    # Contrastamos ausencia vs objetivo con Fisher exacto porque la clase sick es rara (celdas pequeñas).
    print("\nAusencia frente a objetivo (Fisher exacto, bilateral, Holm):")
    sick_rate = 100 * (df[TARGET] == "sick").mean()
    tests = {"alguna analítica": df[["TSH", "T3", "TT4", "T4U", "FTI"]].isna().any(axis=1)}
    tests.update({c: df[c].isna() for c in missing_cols})
    rows = []
    for name, mask in tests.items():
        contingency = pd.crosstab(mask, df[TARGET]).reindex(index=[False, True], columns=["negative", "sick"], fill_value=0)
        p = stats.fisher_exact(contingency)[1]
        rows.append({"ausencia": name, "n": int(mask.sum()), "%sick_ausentes": 100 * (df.loc[mask, TARGET] == "sick").mean(),
                     "%sick_presentes": 100 * (df.loc[~mask, TARGET] == "sick").mean(),
                     "OR_Haldane": haldane_odds_ratio(contingency), "p": p})
    result = pd.DataFrame(rows).set_index("ausencia")
    result["p_Holm"] = holm_correction(result["p"])
    with pd.option_context("display.float_format", "{:.4g}".format):
        print(result.to_string())
    print(f"(% sick global: {sick_rate:.2f})")
    # Comparamos la edad y el origen de las filas con y sin analíticas para razonar si la ausencia es MAR.
    any_lab = tests["alguna analítica"]
    u, p = stats.mannwhitneyu(df.loc[any_lab, "age"].dropna(), df.loc[~any_lab, "age"].dropna())
    print(f"Edad: mediana con analítica ausente={df.loc[any_lab, 'age'].median():.0f}, "
          f"completa={df.loc[~any_lab, 'age'].median():.0f}, Mann-Whitney p={p:.2e}")
    print("referral_source de las filas sin T3:", df.loc[df["T3"].isna(), "referral_source"].value_counts().to_dict())
    print("referral_source global:", df["referral_source"].value_counts().to_dict())
    sex_table = pd.crosstab(df["sex"].isna(), df[TARGET])
    print(f"sex ausente: % sick={100 * (df.loc[df['sex'].isna(), TARGET] == 'sick').mean():.2f}, "
          f"Fisher p={stats.fisher_exact(sex_table)[1]:.4f}")

    # Figura: barras de ausentes por columna y mapa de ausencias por fila.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    shown = missing_counts[missing_counts > 0]
    bars = ax.bar(shown.index, shown.values, color=COLOR_NEG, width=0.6)
    for bar, value in zip(bars, shown.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 20, f"{value}\n({100 * value / len(df):.1f} %)",
                ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_ylim(0, len(df) * 1.15)
    ax.set_title("Valores ausentes por columna")
    ax.set_ylabel("Nº de filas con '?'")
    ax = axes[1]
    lab_cols = missing_cols
    order = np.lexsort([df[lab_cols].isna().astype(int).values[:, k] for k in range(len(lab_cols))][::-1])[::-1]
    sub = indicator[lab_cols].iloc[order[:rows_with_missing]]
    ax.imshow(sub.values, aspect="auto", cmap=ListedColormap(["#f0efec", COLOR_SICK]), interpolation="nearest")
    ax.set_xticks(range(len(lab_cols)), lab_cols, rotation=45)
    ax.set_yticks([0, rows_with_missing - 1], ["1", str(rows_with_missing)])
    ax.set_ylabel("Filas con algún ausente (ordenadas por patrón)")
    ax.set_title(f"Mapa de ausencias sin TBG ({rows_with_missing} de {len(df)} filas)")
    ax.grid(False)
    fig.suptitle("Sick: valores ausentes (TBG falta en el 100 % de las filas)", fontweight="bold")
    save_figure(fig, "missing_values", "missing values per column and missingness map")

    # Figura: tasa de sick según si cada analítica está o no, porque la ausencia es informativa.
    # Omitimos age en la figura porque solo tiene un ausente y su tasa no es interpretable.
    plotted = result.drop(index="age")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    names = list(plotted.index)
    positions = np.arange(len(names))
    ax.bar(positions - 0.2, plotted["%sick_presentes"], width=0.38, color=COLOR_NEG, label="Valor presente")
    ax.bar(positions + 0.2, plotted["%sick_ausentes"], width=0.38, color=COLOR_SICK, label="Valor ausente")
    ax.axhline(sick_rate, color=INK_SECONDARY, linestyle="--", linewidth=1, label=f"Global ({sick_rate:.1f} %)")
    ax.set_xticks(positions, [f"{n}\nn aus.={k}\np={p:.0e}" for n, k, p in
                              zip(names, plotted["n"], plotted["p_Holm"])], fontsize=7.5)
    ax.set_ylabel("% de pacientes sick")
    ax.set_ylim(0, plotted[["%sick_presentes", "%sick_ausentes"]].values.max() * 1.25)
    ax.set_title("Tasa de sick según la ausencia de cada variable (Fisher exacto, p con Holm)")
    ax.legend(loc="upper right", ncol=3)
    save_figure(fig, "missingness_vs_target", "sick rate when each variable is present vs missing")
    return result


def analyse_validity(df):
    # === 4. Valores inválidos y consistencia ===
    section("4. Valores inválidos y consistencia")
    # Revisamos rangos fisiológicos: una edad > 120 o una analítica <= 0 no son posibles.
    for col in CONTINUOUS:
        values = df[col].dropna()
        if len(values):
            print(f"  {col}: min={values.min()}, max={values.max()}, <=0: {int((values <= 0).sum())}")
    print("Edades > 120:", df.loc[df["age"] > 120, ["record_id", "age", TARGET]].to_dict("records"))
    print("Cinco edades más altas:", sorted(df["age"].nlargest(5).tolist(), reverse=True))
    # Comprobamos que el embarazo es coherente con el sexo, porque un hombre embarazado sería una errata.
    print("\nTabla sex x pregnant (NaN incluido):")
    print(pd.crosstab(df["sex"].fillna("NaN"), df["pregnant"]).to_string())
    # FTI se calcula clínicamente como TT4 / T4U; si se cumple, FTI es casi redundante.
    m = df[["TT4", "T4U", "FTI"]].dropna()
    ratio = m["TT4"] / m["T4U"]
    print(f"\nFTI frente a TT4/T4U: Pearson={np.corrcoef(ratio, m['FTI'])[0, 1]:.5f}, "
          f"|dif| mediana={(ratio - m['FTI']).abs().median():.3f}, |dif|<=1 en {100 * ((ratio - m['FTI']).abs() <= 1).mean():.1f} %")
    print("Ausencia de FTI frente a TT4/T4U ausentes:")
    print(pd.crosstab(df["FTI"].isna(), [df["TT4"].isna().rename("TT4_NaN"), df["T4U"].isna().rename("T4U_NaN")]).to_string())


def analyse_duplicates_cardinality(df):
    # === 5. Duplicados, columnas constantes y cardinalidad ===
    section("5. Duplicados, columnas constantes y cardinalidad")
    # Buscamos duplicados con y sin el objetivo; los segundos revelarían etiquetas contradictorias.
    features = [c for c in COLUMNS if c != TARGET]
    dup_all = df.duplicated(subset=COLUMNS, keep=False)
    print(f"Filas duplicadas exactas (sin contar la primera): {int(df.duplicated(subset=COLUMNS).sum())}")
    print(f"Duplicadas en atributos ignorando el objetivo: {int(df.duplicated(subset=features).sum())}")
    print(f"Filas implicadas: {int(dup_all.sum())}, todas con id distinto: {df.loc[dup_all, 'record_id'].is_unique}")
    dup = df[dup_all]
    print(f"De ellas sin ninguna analítica: {int(dup[['TSH', 'T3', 'TT4', 'T4U', 'FTI']].isna().all(axis=1).sum())}; "
          f"clases: {dup[TARGET].value_counts().to_dict()}")
    # Medimos el valor dominante; por encima del 99 % la columna casi no varía.
    rows = []
    for col in COLUMNS:
        counts = df[col].value_counts(normalize=True)
        rows.append({"columna": col, "n_unicos": df[col].nunique(),
                     "valor_dominante": counts.index[0] if len(counts) else "-",
                     "%_dominante": round(100 * counts.iloc[0], 2) if len(counts) else np.nan,
                     "n_minoritario": int(len(df[col].dropna()) - df[col].value_counts().iloc[0]) if len(counts) else 0})
    cardinality = pd.DataFrame(rows).set_index("columna")
    print(cardinality.to_string())
    print(f"Columnas constantes: {cardinality[cardinality.n_unicos <= 1].index.tolist()}")
    print(f"Casi constantes (<20 filas en el nivel minoritario): "
          f"{cardinality[(cardinality.n_unicos == 2) & (cardinality.n_minoritario < 20)].index.tolist()}")
    print("Única fila con hypopituitary = t:", df.loc[df["hypopituitary"] == "t", ["record_id", "age", "sex", TARGET]].to_dict("records"))


def analyse_continuous(df):
    # === 6. Estadística univariante: continuas ===
    section("6. Estadística univariante: continuas")
    # Excluimos la edad imposible (455) para que no distorsione los estadísticos; la limpieza la anula.
    data = df[CONTINUOUS[:-1]].copy()
    data.loc[data["age"] > 120, "age"] = np.nan
    cols = list(data.columns)
    rows = []
    for col in cols:
        x = data[col].dropna()
        q1, q3 = x.quantile([0.25, 0.75])
        rows.append({"col": col, "n": len(x), "media": x.mean(), "mediana": x.median(), "std": x.std(),
                     "min": x.min(), "Q1": q1, "Q3": q3, "max": x.max(), "asimetría": stats.skew(x),
                     "curtosis_exc": stats.kurtosis(x), "asim_log": stats.skew(np.log(x)),
                     "asim_log1p": stats.skew(np.log1p(x))})
    summary = pd.DataFrame(rows).set_index("col")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(summary.to_string())
    # Shapiro-Wilk es potente para n < 5000; D'Agostino K² se basa en asimetría y curtosis.
    print("\nNormalidad:")
    for col in cols:
        x = data[col].dropna()
        print(f"  {col}: Shapiro W={stats.shapiro(x).statistic:.3f} p={stats.shapiro(x).pvalue:.2e} | "
              f"D'Agostino K²={stats.normaltest(x).statistic:.1f} p={stats.normaltest(x).pvalue:.2e}")
    # Usamos la regla IQR (robusta) porque varias analíticas son muy asimétricas; |z| > 3 solo como referencia.
    print("\nAtípicos: regla IQR (1.5·IQR) en escala original y en log, y |z| > 3:")
    for col in cols:
        x = data[col].dropna()
        q1, q3 = x.quantile([0.25, 0.75])
        iqr_out = int(((x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))).sum())
        lx = np.log(x)
        lq1, lq3 = lx.quantile([0.25, 0.75])
        log_out = int(((lx < lq1 - 1.5 * (lq3 - lq1)) | (lx > lq3 + 1.5 * (lq3 - lq1))).sum())
        z_out = int((np.abs(stats.zscore(x)) > 3).sum())
        print(f"  {col}: IQR={iqr_out:3d} ({100 * iqr_out / len(x):4.1f} %), IQR(log)={log_out:3d}, |z|>3={z_out:3d}")
    print(f"Mayores TSH: {sorted(data['TSH'].nlargest(5).tolist(), reverse=True)}; "
          f"menores: {sorted(data['TSH'].nsmallest(5).tolist())}")

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, col in zip(axes.flat, cols):
        x = data[col].dropna()
        ax.hist(x, bins=50, density=True, color=COLOR_NEG, alpha=0.55, edgecolor="white", linewidth=0.4)
        grid = np.linspace(x.min(), x.max(), 400)
        ax.plot(grid, stats.gaussian_kde(x)(grid), color=INK, linewidth=1.5, label="KDE")
        ax.axvline(x.median(), color=COLOR_SICK, linewidth=1.5, linestyle="--", label="Mediana")
        ax.set_title(f"{col} (asimetría {stats.skew(x):.2f}, n = {len(x)})")
        ax.set_xlabel(col)
        ax.set_ylabel("Densidad")
    axes.flat[0].legend()
    fig.suptitle("Distribución de edad y analíticas (sin la edad imposible de 455)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "continuous_histograms", "histograms with KDE of age and lab values")

    # Mostramos TSH en log porque su asimetría es extrema (13.9) y el log la reduce a -0.5.
    # T4U también mejora con log (de 1.23 a -0.02), pero su asimetría es moderada y apenas pierde atípicos IQR.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, transform, name in [(axes[0], np.log1p, "log(1 + TSH)"), (axes[1], np.log, "log(TSH)")]:
        x = transform(data["TSH"].dropna())
        ax.hist(x, bins=50, density=True, color=COLOR_NEG, alpha=0.55, edgecolor="white", linewidth=0.4)
        grid = np.linspace(x.min(), x.max(), 400)
        ax.plot(grid, stats.gaussian_kde(x)(grid), color=INK, linewidth=1.5)
        ax.set_title(f"{name} (asimetría {stats.skew(x):.2f})")
        ax.set_xlabel(name)
        ax.set_ylabel("Densidad")
    fig.suptitle("TSH: log1p frente a log natural", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "tsh_log_transforms", "histograms of TSH under log1p and natural log")
    return summary


def analyse_categorical(df):
    # === 7. Estadística univariante: categóricas ===
    section("7. Estadística univariante: categóricas")
    for col in CATEGORICAL:
        counts = df[col].value_counts(dropna=False)
        pct = (100 * counts / len(df)).round(2)
        print(f"  {col}: " + ", ".join(f"{k}={v} ({p} %)" for k, v, p in zip(counts.index, counts, pct)))


def analyse_target(df):
    # === 8. Balance del objetivo ===
    section("8. Balance del objetivo")
    counts = df[TARGET].value_counts().reindex(["negative", "sick"])
    ratio = counts["negative"] / counts["sick"]
    print(f"Clases: {counts.to_dict()} | sick = {100 * counts['sick'] / len(df):.2f} % | ratio = {ratio:.2f} : 1")
    # Con un 6 % de positivos la exactitud no informa: un clasificador que siempre dice negativo acierta el 94 %.
    print(f"Exactitud del clasificador trivial 'siempre negativo': {100 * counts['negative'] / len(df):.2f} %")
    # Calculamos cuántos sick caerían en una validación de 20 % y en cada pliegue de 5-fold estratificado.
    print(f"sick esperados en un 20 % estratificado: {0.2 * counts['sick']:.1f}; por pliegue 5-fold sobre el 80 %: "
          f"{0.8 * counts['sick'] / 5:.1f}")
    fig, ax = plt.subplots(figsize=(5, 3.6))
    bars = ax.bar([CLASS_LABELS[c] for c in counts.index], counts.values,
                  color=[CLASS_COLORS[c] for c in counts.index], width=0.55)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 25, f"{value} ({100 * value / len(df):.1f} %)",
                ha="center", va="bottom", color=INK)
    ax.set_ylabel("Nº de pacientes")
    ax.set_title(f"Balance del objetivo (ratio {ratio:.1f} : 1)")
    ax.set_ylim(0, counts.max() * 1.15)
    save_figure(fig, "target_balance", "target class balance")


def analyse_continuous_vs_target(df):
    # === 9. Bivariante: continuas frente al objetivo ===
    section("9. Bivariante: continuas frente al objetivo")
    # No son normales (sección 6), así que usamos Mann-Whitney U con la biserial de rangos como efecto.
    data = df.copy()
    data.loc[data["age"] > 120, "age"] = np.nan
    cols = CONTINUOUS[:-1]
    rows = []
    for col in cols:
        neg = data.loc[data[TARGET] == "negative", col].dropna()
        sick = data.loc[data[TARGET] == "sick", col].dropna()
        u, p_mw = stats.mannwhitneyu(sick, neg, alternative="two-sided")
        p_t = stats.ttest_ind(sick, neg, equal_var=False).pvalue
        rows.append({"col": col, "n_sick": len(sick), "mediana_sick": sick.median(), "mediana_neg": neg.median(),
                     "U": u, "p_MW": p_mw, "p_Welch(ref)": p_t, "r_rank_biserial": 2 * u / (len(sick) * len(neg)) - 1})
    result = pd.DataFrame(rows).set_index("col")
    result["p_MW_Holm"] = holm_correction(result["p_MW"])
    with pd.option_context("display.float_format", "{:.4g}".format):
        print(result.to_string())
    print("(r > 0: valores mayores en sick; |r| ~0.1 pequeño, ~0.3 medio, ~0.5 grande)")

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, col in zip(axes.flat, cols):
        groups = [data.loc[data[TARGET] == cls, col].dropna() for cls in ["negative", "sick"]]
        if col == "TSH":
            groups = [np.log(g) for g in groups]
        box = ax.boxplot(groups, widths=0.5, patch_artist=True, showfliers=True,
                         flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.5, "markeredgecolor": INK_SECONDARY},
                         medianprops={"color": INK, "linewidth": 1.5})
        for patch, cls in zip(box["boxes"], ["negative", "sick"]):
            patch.set_facecolor(CLASS_COLORS[cls])
            patch.set_alpha(0.75)
        ax.set_xticks([1, 2], [CLASS_LABELS["negative"], CLASS_LABELS["sick"]])
        ax.set_ylabel("log(TSH)" if col == "TSH" else col)
        row = result.loc[col]
        ax.set_title(f"{col}: r = {row['r_rank_biserial']:.2f}, p Holm = {row['p_MW_Holm']:.1e}")
    fig.suptitle("Edad y analíticas por clase (TSH en log; Mann-Whitney U)", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "continuous_boxplots_by_target", "boxplots of age and lab values by target")
    return result


def analyse_categorical_vs_target(df):
    # === 10. Bivariante: categóricas frente al objetivo ===
    section("10. Bivariante: categóricas frente al objetivo")
    # 2x2: Fisher exacto con odds ratio de Haldane; RxC: chi-cuadrado o permutación si hay esperadas < 5.
    rows = []
    for col in CATEGORICAL:
        sub = df[[col, TARGET]].dropna()
        table = pd.crosstab(sub[col], sub[TARGET]).reindex(columns=["negative", "sick"], fill_value=0)
        if table.shape[0] < 2:
            rows.append({"col": col, "niveles": table.shape[0], "test": "constante"})
            continue
        chi2, p_chi2, dof, expected = stats.chi2_contingency(table, correction=False)
        low_expected = int((expected < 5).sum())
        if table.shape == (2, 2):
            test, p = "Fisher exacto", stats.fisher_exact(table)[1]
            # Orientamos el OR como odds de sick en el nivel 't' (o 'M') frente al otro nivel.
            level_order = ["M", "F"] if col == "sex" else ["t", "f"]
            odds = haldane_odds_ratio(table.reindex(index=level_order)[["sick", "negative"]])
        elif low_expected == 0:
            test, p, odds = "chi-cuadrado", p_chi2, np.nan
        else:
            test, p, odds = "chi² permutación", chi2_permutation_pvalue(sub[col], sub[TARGET]), np.nan
        rates = (100 * table["sick"] / table.sum(axis=1)).round(1).to_dict()
        rows.append({"col": col, "niveles": table.shape[0], "celdas_esp<5": low_expected, "test": test, "p": p,
                     "OR_Haldane": odds, "V_Cramer": cramers_v(sub[col], sub[TARGET]), "%sick_por_nivel": rates})
    result = pd.DataFrame(rows).set_index("col")
    tested = result["p"].notna()
    result.loc[tested, "p_Holm"] = holm_correction(result.loc[tested, "p"])
    with pd.option_context("display.float_format", "{:.4g}".format, "display.width", 200):
        print(result.drop(columns="%sick_por_nivel").to_string())
    print("% sick por nivel:")
    for col, rates in result["%sick_por_nivel"].dropna().items():
        print(f"  {col}: {rates}")
    print("(OR > 1: más sick en el nivel 't'/'M'; V ~0.1 pequeño, ~0.3 medio, ~0.5 grande)")

    # Figura: prevalencia de cada flag y tasa de sick en ella, porque con 6 % de positivos los recuentos engañan.
    binary = [c for c in ["sex"] + BINARY_TF if result.loc[c, "test"] != "constante"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), gridspec_kw={"width_ratios": [1, 1.2]})
    positive_level = {c: ("M" if c == "sex" else "t") for c in binary}
    prevalence = [100 * (df[c] == positive_level[c]).sum() / df[c].notna().sum() for c in binary]
    labels = [f"{c} = {positive_level[c]}" for c in binary]
    y = np.arange(len(binary))[::-1]
    axes[0].barh(y, prevalence, color=COLOR_NEG, height=0.6)
    axes[0].set_yticks(y, labels, fontsize=7.5)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("% de pacientes con el nivel (escala log)")
    axes[0].set_title("Prevalencia del nivel 't' / 'M'")
    rate_pos = [100 * (df.loc[df[c] == positive_level[c], TARGET] == "sick").mean() for c in binary]
    rate_neg = [100 * (df.loc[(df[c] != positive_level[c]) & df[c].notna(), TARGET] == "sick").mean() for c in binary]
    axes[1].barh(y + 0.2, rate_pos, height=0.38, color=COLOR_SICK, label="Nivel 't' / 'M'")
    axes[1].barh(y - 0.2, rate_neg, height=0.38, color=COLOR_NEG, label="Nivel 'f' / 'F'")
    for k, c in enumerate(binary):
        p_holm = result.loc[c, "p_Holm"]
        axes[1].text(min(max(rate_pos[k], rate_neg[k]), 24) + 0.5, y[k], f"p={p_holm:.0e}" if p_holm < 0.05 else "n.s.",
                     va="center", fontsize=6.5, color=INK_SECONDARY)
    axes[1].axvline(100 * (df[TARGET] == "sick").mean(), color=INK_SECONDARY, linestyle="--", linewidth=1)
    # Cortamos el eje en 30 % para que el 100 % de hypopituitary (n = 1) no aplaste el resto.
    x_max = 30
    for k, rate in enumerate(rate_pos):
        if rate > x_max:
            axes[1].text(x_max - 0.3, y[k] + 0.2, f"{rate:.0f} % (n = {int((df[binary[k]] == positive_level[binary[k]]).sum())}) →",
                         ha="right", va="center", fontsize=6.5, color="white")
    axes[1].set_xlim(0, x_max)
    axes[1].set_yticks(y, [""] * len(binary))
    axes[1].set_xlabel("% de sick")
    axes[1].set_title("Tasa de sick por nivel (p de Fisher con Holm)")
    axes[1].legend(loc="lower right")
    fig.suptitle("Variables binarias frente al objetivo", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "binary_by_target", "prevalence and sick rate of binary columns")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    table = pd.crosstab(df["referral_source"], df[TARGET]).reindex(columns=["negative", "sick"])
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
    pos = np.arange(len(table))
    for offset, cls in [(-0.2, "negative"), (0.2, "sick")]:
        ax.bar(pos + offset, table[cls], width=0.38, color=CLASS_COLORS[cls], label=CLASS_LABELS[cls])
    rates = 100 * table["sick"] / table.sum(axis=1)
    ax.set_xticks(pos, [f"{lvl}\n{r:.1f} % sick" for lvl, r in zip(table.index, rates)], fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylabel("Nº de pacientes (escala log)")
    ax.set_title(f"referral_source por clase ({result.loc['referral_source', 'test']}, "
                 f"V = {result.loc['referral_source', 'V_Cramer']:.2f})")
    ax.legend()
    save_figure(fig, "referral_source_by_target", "referral source counts and sick rate by level")
    return result


def analyse_correlations(df):
    # === 11. Correlaciones entre continuas y colinealidad ===
    section("11. Correlaciones entre continuas y colinealidad")
    data = df[CONTINUOUS[:-1]].copy()
    data.loc[data["age"] > 120, "age"] = np.nan
    cols = list(data.columns)
    # Preferimos Spearman porque no hay normalidad y hay atípicos; Pearson queda como referencia.
    pearson = data.corr(method="pearson")
    spearman = data.corr(method="spearman")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print("Pearson:\n" + pearson.to_string())
        print("Spearman:\n" + spearman.to_string())
    upper = spearman.where(np.triu(np.ones(spearman.shape, dtype=bool), k=1)).stack()
    print("Pares con |Spearman| > 0.5:", upper[upper.abs() > 0.5].round(3).to_dict())
    # VIF sobre filas completas (TSH en log) para ver colinealidad multivariante, no solo por pares.
    complete = data.dropna().copy()
    complete["TSH"] = np.log(complete["TSH"])
    vif = {}
    for col in cols:
        yv = complete[col].values
        X = np.column_stack([np.ones(len(complete)), complete.drop(columns=col).values])
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        r2 = 1 - ((yv - X @ beta) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        vif[col] = 1 / (1 - r2)
    print(f"VIF (n = {len(complete)}):", {k: round(float(v), 2) for k, v in vif.items()}, "(> 5 preocupante, > 10 grave)")
    # Repetimos sin FTI porque sospechamos que es casi TT4/T4U.
    reduced = complete.drop(columns="FTI")
    vif_reduced = {}
    for col in reduced.columns:
        yv = reduced[col].values
        X = np.column_stack([np.ones(len(reduced)), reduced.drop(columns=col).values])
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        vif_reduced[col] = 1 / (1 - (1 - ((yv - X @ beta) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()))
    print("VIF sin FTI:", {k: round(float(v), 2) for k, v in vif_reduced.items()})
    lr = complete[["TT4", "T4U", "FTI"]]
    print(f"Spearman log(FTI) frente a log(TT4) - log(T4U): "
          f"{stats.spearmanr(np.log(lr['FTI']), np.log(lr['TT4']) - np.log(lr['T4U'])).statistic:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, matrix, name in [(axes[0], pearson, "Pearson"), (axes[1], spearman, "Spearman")]:
        im = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)), cols)
        ax.set_yticks(range(len(cols)), cols)
        for i in range(len(cols)):
            for j in range(len(cols)):
                value = matrix.values[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(value) > 0.6 else INK)
        ax.set_title(f"Correlación de {name}")
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.85, label="Coeficiente")
    fig.suptitle("Correlaciones entre edad y analíticas", fontweight="bold")
    save_figure(fig, "correlation_continuous", "Pearson and Spearman correlation heatmaps of continuous columns")
    return spearman, vif


def analyse_associations(df):
    # === 12. Asociaciones entre todas las variables ===
    section("12. Asociaciones entre todas las variables")
    # Medidas en [0, 1]: V de Cramér (cat-cat), eta sobre rangos (cat-cont) y |Spearman| (cont-cont).
    data = df.copy()
    data.loc[data["age"] > 120, "age"] = np.nan
    # Quitamos TBG y TBG_measured porque son constantes (vacía y siempre 'f') y no admiten asociación.
    cols = [c for c in COLUMNS if c not in ("TBG", "TBG_measured")]
    categorical_like = CATEGORICAL + [TARGET]
    matrix = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if a in categorical_like and b in categorical_like:
                value = cramers_v(data[a], data[b])
            elif a in categorical_like:
                value = correlation_ratio(data[a], data[b])
            elif b in categorical_like:
                value = correlation_ratio(data[b], data[a])
            else:
                value = abs(data[[a, b]].corr(method="spearman").iloc[0, 1])
            matrix.loc[a, b] = matrix.loc[b, a] = value
    upper = matrix.where(np.triu(np.ones(matrix.shape, dtype=bool), k=1)).stack().sort_values(ascending=False)
    print("Pares más asociados:")
    print(upper.head(15).round(3).to_string())
    print("Asociación de cada variable con el objetivo (top 10):")
    print(matrix[TARGET].drop(TARGET).sort_values(ascending=False).head(10).round(3).to_string())

    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(matrix.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(cols)), cols, fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            value = matrix.values[i, j]
            if i != j and value >= 0.1:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5,
                        color="white" if value > 0.55 else INK)
    ax.set_title("Asociación entre variables (solo se rotulan valores >= 0.10)\n"
                 "V de Cramér cat-cat, eta sobre rangos cat-cont, |Spearman| cont-cont")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Fuerza de asociación [0, 1]")
    save_figure(fig, "association_matrix", "mixed-type association matrix of all columns")
    return matrix


def analyse_scatter_matrix(df):
    # === 13. Matriz de dispersión de las analíticas por clase ===
    section("13. Matriz de dispersión de las analíticas por clase")
    # Usamos filas completas y log(TSH) para ver si alguna pareja separa las clases.
    cols = ["TSH", "T3", "TT4", "T4U", "FTI"]
    data = df[cols + [TARGET]].dropna().copy()
    data["TSH"] = np.log(data["TSH"])
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(11, 11))
    for i, row_col in enumerate(cols):
        for j, col_col in enumerate(cols):
            ax = axes[i, j]
            if i == j:
                for cls in ["negative", "sick"]:
                    ax.hist(data.loc[data[TARGET] == cls, row_col], bins=30, alpha=0.55,
                            color=CLASS_COLORS[cls], density=True)
                # Ocultamos el eje y de la diagonal porque es densidad, no la escala de la fila.
                ax.set_yticks([])
            else:
                # Dibujamos sick encima para que la clase minoritaria no quede tapada.
                for cls, size in [("negative", 3), ("sick", 6)]:
                    sub = data[data[TARGET] == cls]
                    ax.scatter(sub[col_col], sub[row_col], s=size, alpha=0.5, color=CLASS_COLORS[cls], linewidths=0)
            label = lambda c: "log(TSH)" if c == "TSH" else c
            if i == n - 1:
                ax.set_xlabel(label(col_col))
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(label(row_col))
            else:
                ax.set_yticklabels([])
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=CLASS_COLORS[c], label=CLASS_LABELS[c])
               for c in ["negative", "sick"]]
    fig.legend(handles=handles, loc="upper right")
    fig.suptitle(f"Matriz de dispersión de las analíticas (n = {len(data)} filas completas)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.95, 0.97))
    save_figure(fig, "scatter_matrix", "scatter matrix of lab values by target")
    print(f"Filas usadas: {len(data)} (sick: {int((data[TARGET] == 'sick').sum())})")


def analyse_mutual_information(df):
    # === 14. Información mutua con el objetivo ===
    section("14. Información mutua con el objetivo")
    # La IM capta dependencias no lineales y da una referencia de importancia para comparar con XAI.
    # La calculamos con todas las filas: ausente se codifica como -1 en categóricas y la mediana en continuas.
    data = df[[c for c in COLUMNS if c not in ("TBG", "TBG_measured")]].copy()
    data.loc[data["age"] > 120, "age"] = np.nan
    X = data.drop(columns=TARGET)
    discrete = []
    for col in X.columns:
        if col in CATEGORICAL:
            X[col] = pd.Categorical(X[col], categories=DOCUMENTED_LEVELS[col]).codes
            discrete.append(True)
        else:
            X[col] = X[col].fillna(X[col].median())
            discrete.append(False)
    y = (data[TARGET] == "sick").astype(int)
    mi = pd.Series(mutual_info_classif(X, y, discrete_features=discrete, random_state=SEED),
                   index=X.columns).sort_values(ascending=False)
    print(f"Entropía del objetivo: {stats.entropy(y.value_counts(normalize=True)):.4f} nats (cota superior de la IM)")
    print(mi.round(4).to_string())

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.barh(mi.index[::-1], mi.values[::-1], color=COLOR_NEG, height=0.6)
    for k, v in enumerate(mi.values[::-1]):
        ax.text(v + 0.001, k, f"{v:.3f}", va="center", fontsize=7, color=INK)
    ax.set_xlabel("Información mutua con el objetivo (nats)")
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
    run_cleanup()
