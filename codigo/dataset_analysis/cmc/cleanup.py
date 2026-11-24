"""Limpieza del dataset Contraceptive Method Choice (UCI, cmc) y preprocesado ajustado en TRAIN.

Ejecución: python cleanup.py (desde cualquier carpeta); parte del CSV crudo y escribe cmc_cleaned.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_PATH = SCRIPT_DIR / ".." / ".." / "datasets" / "cmc" / "cmc.csv"
CLEANED_PATH = SCRIPT_DIR / "cmc_cleaned.csv"


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# === 15. Limpieza determinista (sin estadísticos aprendidos) ===
CLEAN_COLUMNS = ["wife_age", "wife_edu", "husband_edu", "num_children", "wife_religion", "wife_working",
                 "husband_occupation", "standard_of_living_index", "media_exposure", "contraceptive_method"]
CLEAN_TARGET = "contraceptive_method"
# Mantenemos los duplicados exactos porque se comportan como mujeres distintas con el mismo perfil.
# Si ponemos True los eliminamos sin estadísticos aprendidos; por defecto conservamos las 1473 filas.
DROP_EXACT_DUPLICATES = False
CLEAN_CODES = {
    "wife_edu": [1, 2, 3, 4],
    "husband_edu": [1, 2, 3, 4],
    "wife_religion": [0, 1],
    "wife_working": [0, 1],
    "husband_occupation": [1, 2, 3, 4],
    "standard_of_living_index": [1, 2, 3, 4],
    "media_exposure": [0, 1],
}
CLEAN_NUMERIC = ["wife_age", "num_children"]
# Las ordinales (1=bajo..4=alto en cmc.names) conservan su código para no perder el orden.
CLEAN_ORDINAL = ["wife_edu", "husband_edu", "standard_of_living_index"]
CLEAN_BINARY = ["wife_religion", "wife_working", "media_exposure"]
# husband_occupation no tiene orden documentado: one-hot con sus 4 códigos fijos.
CLEAN_NOMINAL = {"husband_occupation": [1, 2, 3, 4]}
# La encuesta (NICPS 1987) recoge mujeres casadas en edad fértil, de 15 a 49 años.
CLEAN_AGE_RANGE = (15, 49)
# Pasamos el objetivo a 0..2 con un mapa fijo, como esperan las funciones de pérdida categóricas.
CLEAN_TARGET_MAP = {1: 0, 2: 1, 3: 2}
CLEAN_TARGET_NAMES = {0: "no_use", 1: "long_term", 2: "short_term"}


def load_raw(path):
    """Leemos cmc.csv crudo, sin cabecera, con los nombres de columna de main.ipynb."""
    # index_col=False evita que una coma final en cada fila convierta la primera columna en índice.
    return pd.read_csv(path, header=None, names=CLEAN_COLUMNS, dtype=str, keep_default_na=False, index_col=False)


def clean_deterministic(raw, drop_exact_duplicates=DROP_EXACT_DUPLICATES):
    """Limpiamos sin estadísticos aprendidos: tipos, códigos fuera de esquema a NaN y objetivo 0..2."""
    # Quitamos espacios y pasamos a enteros que admiten NaN; un texto, un decimal no entero o un infinito quedan como NaN.
    df = raw.apply(lambda s: s.str.strip())
    for col in df.columns:
        values = pd.to_numeric(df[col], errors="coerce").astype(float)
        values[~np.isfinite(values) | (values % 1 != 0) | (values.abs() > 2 ** 31)] = np.nan
        df[col] = values.astype("Int64")
    # Anulamos códigos fuera de cmc.names y edades fuera de la población encuestada (hoy no hay ninguno).
    for col, codes in CLEAN_CODES.items():
        invalid = df[col].notna() & ~df[col].isin(codes)
        if invalid.any():
            print(f"  {col}: {int(invalid.sum())} códigos fuera de la documentación -> NaN")
            df.loc[invalid, col] = pd.NA
    low, high = CLEAN_AGE_RANGE
    invalid_age = df["wife_age"].notna() & ~df["wife_age"].between(low, high)
    invalid_children = df["num_children"].notna() & (df["num_children"] < 0)
    df.loc[invalid_age, "wife_age"] = pd.NA
    df.loc[invalid_children, "num_children"] = pd.NA
    print(f"  Edades fuera de [{low}, {high}] -> NaN: {int(invalid_age.sum())}; hijos negativos -> NaN: "
          f"{int(invalid_children.sum())}")
    # Una fila sin objetivo válido no sirve para entrenar ni evaluar; la descartamos.
    df[CLEAN_TARGET] = df[CLEAN_TARGET].map(CLEAN_TARGET_MAP).astype("Int64")
    no_target = df[CLEAN_TARGET].isna()
    if no_target.any():
        print(f"  Filas sin objetivo válido eliminadas: {int(no_target.sum())}")
        df = df[~no_target]
    if drop_exact_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        print(f"  Duplicados exactos eliminados: {before - len(df)}")
    return df.reset_index(drop=True)


def one_hot_encode(df):
    """Aplicamos one-hot a la nominal con sus códigos documentados (fijos, sin ajustar a los datos)."""
    out = df.copy()
    for col, levels in CLEAN_NOMINAL.items():
        if col not in out.columns:
            continue
        for level in levels:
            out[f"{col}_{level}"] = (out[col] == level).astype(float)
            # Dejamos NaN en todo el bloque si el valor falta, para no confundir ausente con "ningún nivel".
            out.loc[out[col].isna(), f"{col}_{level}"] = np.nan
        out = out.drop(columns=col)
    return out


# === 16. Preprocesado ajustado en TRAIN (no se guarda en cmc_cleaned.csv) ===
class CmcPreprocessor:
    """Imputamos, agrupamos niveles raros, recortamos y codificamos; se ajusta en TRAIN y se aplica a cualquier partición.

    Dejamos el escalado a TINTOlib: su transformer se ajusta en fit() con TRAIN y se reutiliza en transform().
    """

    def __init__(self, rare_min_count=0, clip_quantiles=None, one_hot=True):
        # rare_min_count = 0 desactiva la agrupación; clip_quantiles = (0.0, 0.99) recortaría num_children.
        if rare_min_count > 0 and not one_hot:
            raise ValueError("La agrupación de niveles raros solo se aplica con one_hot=True")
        self.rare_min_count = rare_min_count
        self.clip_quantiles = clip_quantiles
        self.one_hot = one_hot

    def fit(self, train):
        features = train.drop(columns=CLEAN_TARGET, errors="ignore")
        # Fallamos con un mensaje claro si una columna llega vacía a TRAIN, porque no habría valor que imputar.
        empty = [c for c in CLEAN_NUMERIC + CLEAN_ORDINAL + CLEAN_BINARY + list(CLEAN_NOMINAL) if features[c].isna().all()]
        if empty:
            raise ValueError(f"Columnas sin ningún valor en TRAIN: {empty}")
        # Imputamos numéricas y ordinales con la mediana: respeta el orden y, redondeada, cae en un código válido.
        self.medians_ = {c: float(features[c].median()) for c in CLEAN_NUMERIC + CLEAN_ORDINAL}
        # Imputamos binarias y nominal con la moda de TRAIN, el valor más probable sin usar la etiqueta.
        self.modes_ = {c: int(features[c].mode(dropna=True).iloc[0]) for c in CLEAN_BINARY + list(CLEAN_NOMINAL)}
        self.rare_levels_ = {}
        if self.rare_min_count > 0:
            for col, levels in CLEAN_NOMINAL.items():
                counts = features[col].value_counts()
                # Incluimos los códigos que no aparecen en TRAIN, para que no queden columnas vacías solo en train.
                unseen = [level for level in levels if level not in counts.index]
                self.rare_levels_[col] = sorted(int(v) for v in counts[counts < self.rare_min_count].index) + unseen
        self.clip_bounds_ = {}
        if self.clip_quantiles is not None:
            low, high = self.clip_quantiles
            for col in CLEAN_NUMERIC:
                self.clip_bounds_[col] = (float(features[col].quantile(low)), float(features[col].quantile(high)))
        return self

    def transform(self, df):
        out = df.copy()
        for col, value in self.medians_.items():
            # Redondeamos la mediana hacia arriba en los empates (2.5 -> 3), sin el redondeo bancario de round().
            out[col] = out[col].astype(float).fillna(float(np.floor(value + 0.5)))
        for col, value in self.modes_.items():
            out[col] = out[col].astype(float).fillna(value)
        for col, (low, high) in self.clip_bounds_.items():
            out[col] = out[col].clip(low, high)
        other = {}
        for col, rare in self.rare_levels_.items():
            if rare:
                other[col] = out[col].isin(rare).astype(float)
        if self.one_hot:
            out = one_hot_encode(out)
            # Sustituimos los niveles raros por una única columna 'other', igual en train y test.
            for col, indicator in other.items():
                out = out.drop(columns=[f"{col}_{level}" for level in self.rare_levels_[col]])
                out[f"{col}_other"] = indicator
        for col in [c for c in out.columns if c != CLEAN_TARGET]:
            out[col] = out[col].astype(float)
        # Dejamos el objetivo en la última columna, que es la que TINTOlib toma como etiqueta.
        if CLEAN_TARGET in out.columns:
            out = out[[c for c in out.columns if c != CLEAN_TARGET] + [CLEAN_TARGET]]
        return out

    def fit_transform(self, train):
        return self.fit(train).transform(train)


def run_cleanup():
    section("15. Limpieza determinista")
    raw = load_raw(RAW_PATH)
    print(f"Crudo: {raw.shape}")
    cleaned = clean_deterministic(raw)
    print(f"Limpio: {cleaned.shape}; columnas: {list(cleaned.columns)}")
    print("Cambios: tipos enteros; códigos fuera de cmc.names -> NaN (0 casos); objetivo 1/2/3 -> 0/1/2 "
          f"{CLEAN_TARGET_NAMES}; ordinales y binarias con sus códigos; husband_occupation como código")
    print(f"Ausentes que quedan: {int(cleaned.isna().sum().sum())}; duplicados exactos que se conservan: "
          f"{int(cleaned.duplicated().sum())}")
    print(f"Objetivo: { {int(k): int(v) for k, v in cleaned[CLEAN_TARGET].value_counts().sort_index().items()} }")
    cleaned.to_csv(CLEANED_PATH, index=False)
    print(f"Guardado {CLEANED_PATH.name} ({len(cleaned)} filas x {cleaned.shape[1]} columnas)")

    section("16. Demostración del preprocesado ajustado en TRAIN")
    # Releemos el CSV limpio para demostrar que el preprocesado funciona desde el fichero entregado.
    data = pd.read_csv(CLEANED_PATH)
    # Estratificamos por el objetivo para conservar 42.7/22.6/34.7; el 80/20 es solo para la demostración.
    train, test = train_test_split(data, test_size=0.2, stratify=data[CLEAN_TARGET], random_state=SEED)
    preprocessor = CmcPreprocessor().fit(train)
    train_ready = preprocessor.transform(train)
    test_ready = preprocessor.transform(test)
    print(f"Train {train.shape} -> {train_ready.shape}; Test {test.shape} -> {test_ready.shape}")
    print(f"Columnas: {list(train_ready.columns)}")
    print(f"Medianas aprendidas en TRAIN: {preprocessor.medians_}")
    print(f"Modas aprendidas en TRAIN: {preprocessor.modes_}")
    print(f"NaN tras transformar: train={int(train_ready.isna().sum().sum())}, test={int(test_ready.isna().sum().sum())}")
    print(f"Columnas iguales en train y test: {list(train_ready.columns) == list(test_ready.columns)}")
    for name, part in [("train", train), ("test", test)]:
        shares = part[CLEAN_TARGET].value_counts(normalize=True).sort_index().round(3).to_dict()
        print(f"Proporción de clases en {name}: {shares}")
    # Medimos cuántas filas de test tienen una copia exacta en train: es el sesgo optimista de conservarlas.
    train_rows = set(map(tuple, train.values))
    test_in_train = sum(tuple(r) in train_rows for r in test.values)
    features = [c for c in data.columns if c != CLEAN_TARGET]
    train_profiles = set(map(tuple, train[features].values))
    profile_in_train = sum(tuple(r) in train_profiles for r in test[features].values)
    print(f"Filas de test con copia exacta en train: {test_in_train} ({100 * test_in_train / len(test):.1f} %); "
          f"con el mismo perfil de atributos: {profile_in_train} ({100 * profile_in_train / len(test):.1f} %)")
    rare_demo = CmcPreprocessor(rare_min_count=30, clip_quantiles=(0.0, 0.99)).fit(train)
    print(f"Ejemplo con agrupación de niveles raros (<30 en TRAIN): {rare_demo.rare_levels_}, "
          f"recorte p99: {rare_demo.clip_bounds_} -> {rare_demo.transform(test).shape[1]} columnas")


if __name__ == "__main__":
    run_cleanup()
