"""Limpieza del dataset Sick (Thyroid Disease, UCI) y preprocesado ajustado en TRAIN.

Ejecución: python cleanup.py (desde cualquier carpeta); une sick.data y sick.test crudos y escribe sick_cleaned.csv.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / ".." / ".." / "datasets" / "sick"
RAW_PATHS = [DATA_DIR / "sick.data", DATA_DIR / "sick.test"]
CLEANED_PATH = SCRIPT_DIR / "sick_cleaned.csv"


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# === 15. Limpieza determinista (sin estadísticos aprendidos ni etiqueta) ===
# Nombres de codigo/main.ipynb (celda 12), extraídos de sick.names.
CLEAN_COLUMNS = ["age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication", "sick",
                 "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid", "query_hyperthyroid",
                 "lithium", "goitre", "tumor", "hypopituitary", "psych", "TSH_measured", "TSH", "T3_measured",
                 "T3", "TT4_measured", "TT4", "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured",
                 "TBG", "referral_source", "label"]
CLEAN_TARGET = "label"
CLEAN_LABS = ["TSH", "T3", "TT4", "T4U", "FTI"]
CLEAN_CONTINUOUS = ["age"] + CLEAN_LABS
CLEAN_TF = ["on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication", "sick", "pregnant",
            "thyroid_surgery", "I131_treatment", "query_hypothyroid", "query_hyperthyroid", "lithium",
            "goitre", "tumor", "psych"] + [f"{lab}_measured" for lab in CLEAN_LABS]
CLEAN_BINARY_MAPS = {**{c: {"f": 0, "t": 1} for c in CLEAN_TF}, "sex": {"F": 0, "M": 1},
                     CLEAN_TARGET: {"negative": 0, "sick": 1}}
# Quitamos WEST porque está documentado pero no aparece y daría una columna one-hot constante.
CLEAN_NOMINAL = {"referral_source": ["other", "SVI", "SVHC", "STMW", "SVHD"]}
# TBG está vacía (100 %) y TBG_measured es siempre 'f'; hypopituitary tiene un único 't' en 3772 filas.
DROP_COLUMNS = ["TBG", "TBG_measured", "hypopituitary"]
MAX_VALID_AGE = 120
# Aplicamos log natural solo a TSH, cuya asimetría extrema (13.9) baja a -0.5; en T3, TT4 y FTI el log la empeora.
# El log también simetriza T4U (1.23 a -0.02) y log1p mejora T3 (1.73 a -0.29), pero sus colas son moderadas.
# La lista se eligió en el EDA sobre las 3772 filas y queda fija (no se reajusta por partición); admite añadir T3 y T4U.
LOG_COLUMNS = ["TSH"]


def load_raw(paths):
    """Leemos los ficheros de UCI, quitamos el sufijo `.|<id>`, los unimos en orden y pasamos `?` a NaN."""
    rows = []
    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"No encontramos {path}: falta codigo/datasets/sick")
        for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
            match = re.search(r"\.\|\d+$", line)
            if not match:
                raise ValueError(f"{Path(path).name}, línea {number}: sin el sufijo .|<id>")
            rows.append(line[:match.start()].split(","))
    raw = pd.DataFrame(rows, columns=CLEAN_COLUMNS)
    # Convertimos los `?` a NaN porque es el marcador de ausente de UCI.
    return raw.mask(raw == "?")


def clean_deterministic(raw):
    """Limpiamos sin estadísticos aprendidos: tipos, inválidos a NaN, reglas fijas y codificación 0/1."""
    df = raw.copy()
    # Quitamos espacios por si una copia futura de los ficheros los trae; hoy no hay ninguno.
    for col in df.columns:
        df[col] = df[col].where(df[col].isna(), df[col].astype(str).str.strip())
    for col in CLEAN_CONTINUOUS + ["TBG"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    # Anulamos niveles no admitidos para que no aparezcan categorías inventadas (hoy no hay ninguno).
    # WEST está en sick.names pero no aparece; si apareciera, también pasaría a NaN y se imputaría en TRAIN.
    allowed = {**{c: list(m) for c, m in CLEAN_BINARY_MAPS.items()}, **CLEAN_NOMINAL, "TBG_measured": ["f", "t"],
               "hypopituitary": ["f", "t"]}
    for col, levels in allowed.items():
        invalid = df[col].notna() & ~df[col].isin(levels)
        if invalid.any():
            print(f"  {col}: {int(invalid.sum())} valores fuera de los niveles admitidos -> NaN")
            df.loc[invalid, col] = np.nan
    # Anulamos edades imposibles (> 120) y analíticas <= 0, que no son fisiológicas.
    bad_age = df["age"] > MAX_VALID_AGE
    print(f"  age > {MAX_VALID_AGE}: {int(bad_age.sum())} -> NaN ({df.loc[bad_age, 'age'].tolist()})")
    df.loc[bad_age, "age"] = np.nan
    for col in CLEAN_LABS:
        bad = df[col] <= 0
        if bad.any():
            print(f"  {col} <= 0: {int(bad.sum())} -> NaN")
            df.loc[bad, col] = np.nan
    # Hacemos cumplir que *_measured = f implica medida ausente y viceversa (hoy coincide al 100 %).
    for lab in CLEAN_LABS:
        flag = f"{lab}_measured"
        if not ((df[flag] == "f") == df[lab].isna()).all():
            raise ValueError(f"{flag} no coincide con la ausencia de {lab}")
    # Si sex falta y pregnant = t, el sexo solo puede ser F: es una deducción, no una imputación estadística.
    deduced = df["sex"].isna() & (df["pregnant"] == "t")
    print(f"  sex deducido como F por pregnant = t: {int(deduced.sum())}")
    df.loc[deduced, "sex"] = "F"
    df = df.drop(columns=DROP_COLUMNS)
    # Codificamos binarias y objetivo como 0/1 con un mapa fijo (1 = sick), sin depender de los datos.
    for col, mapping in CLEAN_BINARY_MAPS.items():
        df[col] = df[col].map(mapping).astype("Int64")
    df["age"] = df["age"].round().astype("Int64")
    return df


def one_hot_encode(df):
    """Aplicamos one-hot a las nominales con los niveles documentados (fijos, sin ajustar a los datos)."""
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


# === 16. Preprocesado ajustado en TRAIN (no se guarda en sick_cleaned.csv) ===
class SickPreprocessor:
    """Imputamos, recortamos opcionalmente y transformamos; se ajusta en TRAIN y se aplica a cualquier partición.

    Dejamos el escalado a TINTOlib: su transformer se ajusta en fit() con TRAIN y se reutiliza en transform().
    """

    def __init__(self, log_columns=LOG_COLUMNS, clip_quantiles=None, one_hot=True):
        # clip_quantiles = (0.01, 0.99) activaría la winsorización; por defecto no recortamos.
        self.log_columns = list(log_columns)
        self.clip_quantiles = clip_quantiles
        self.one_hot = one_hot

    def fit(self, train):
        features = train.drop(columns=CLEAN_TARGET, errors="ignore")
        # Fallamos pronto si una columna está vacía en TRAIN, porque no habría mediana ni moda con que imputar.
        empty = [c for c in features.columns if features[c].isna().all()]
        if empty:
            raise ValueError(f"Columnas sin ningún valor en TRAIN: {empty}")
        # Imputamos con la mediana de TRAIN: robusta a las colas; el flag *_measured conserva que faltaba.
        self.medians_ = {c: float(features[c].median()) for c in CLEAN_CONTINUOUS}
        # Guardamos la moda de TRAIN de todas las categóricas, aunque no falten en TRAIN, por si faltan en test.
        categorical = [c for c in features.columns if c not in CLEAN_CONTINUOUS]
        self.modes_ = {c: features[c].mode(dropna=True).iloc[0] for c in categorical}
        # Pasamos a int las modas de columnas 0/1, que pandas lee como float por sus NaN.
        self.modes_ = {c: (int(v) if isinstance(v, (int, float, np.number)) else v) for c, v in self.modes_.items()}
        self.clip_bounds_ = {}
        if self.clip_quantiles is not None:
            low, high = self.clip_quantiles
            for col in CLEAN_CONTINUOUS:
                self.clip_bounds_[col] = (float(features[col].quantile(low)), float(features[col].quantile(high)))
        return self

    def transform(self, df):
        out = df.copy()
        for col, value in self.medians_.items():
            out[col] = out[col].astype(float).fillna(value)
        for col, value in self.modes_.items():
            out[col] = out[col].fillna(value)
        for col, (low, high) in self.clip_bounds_.items():
            out[col] = out[col].clip(low, high)
        for col in self.log_columns:
            # Esperamos datos de clean_deterministic, que ya anula los valores <= 0; si llegan, fallamos.
            if (out[col] <= 0).any():
                raise ValueError(f"{col} tiene valores <= 0: no admite log")
            out[col] = np.log(out[col].astype(float))
        if self.one_hot:
            out = one_hot_encode(out)
        for col in [c for c in out.columns if c != CLEAN_TARGET]:
            out[col] = out[col].astype(float)
        return out

    def fit_transform(self, train):
        return self.fit(train).transform(train)


def run_cleanup():
    section("15. Limpieza determinista")
    raw = load_raw(RAW_PATHS)
    print(f"Crudo (sick.data + sick.test sin sufijo): {raw.shape}, ausentes totales: {int(raw.isna().sum().sum())}")
    cleaned = clean_deterministic(raw)
    print(f"Limpio: {cleaned.shape}; columnas: {list(cleaned.columns)}")
    print("Cambios: sufijo .|id quitado; '?' -> NaN; age entera y > 120 -> NaN; analíticas float; "
          "binarias, sex y objetivo -> 0/1; TBG, TBG_measured e hypopituitary eliminadas; referral_source como texto")
    print(f"Ausentes que quedan por columna: {cleaned.isna().sum()[cleaned.isna().sum() > 0].to_dict()}")
    print(f"Objetivo (1 = sick): { {int(k): int(v) for k, v in cleaned[CLEAN_TARGET].value_counts().sort_index().items()} }")
    cleaned.to_csv(CLEANED_PATH, index=False)
    print(f"Guardado {CLEANED_PATH.name} ({len(cleaned)} filas x {cleaned.shape[1]} columnas)")

    section("16. Demostración del preprocesado ajustado en TRAIN")
    # Releemos el CSV limpio para demostrar que el preprocesado funciona desde el fichero entregado.
    data = pd.read_csv(CLEANED_PATH)
    # Estratificamos por el objetivo porque con un 6 % de sick un corte aleatorio podría desequilibrar las particiones.
    # El 80/20 es solo para la demostración; el TEST de la tesis se separa después, también estratificado.
    train, test = train_test_split(data, test_size=0.2, stratify=data[CLEAN_TARGET], random_state=SEED)
    preprocessor = SickPreprocessor().fit(train)
    train_ready = preprocessor.transform(train)
    test_ready = preprocessor.transform(test)
    print(f"Train {train.shape} -> {train_ready.shape}; Test {test.shape} -> {test_ready.shape}")
    print(f"Medianas aprendidas en TRAIN: {preprocessor.medians_}")
    print(f"Modas aprendidas en TRAIN: {preprocessor.modes_}")
    full_medians = {c: float(data[c].median()) for c in CLEAN_CONTINUOUS}
    print(f"(Medianas del dataset completo, NO usadas: {full_medians})")
    print(f"NaN tras transformar: train={int(train_ready.isna().sum().sum())}, test={int(test_ready.isna().sum().sum())}")
    print(f"Columnas iguales en train y test: {list(train_ready.columns) == list(test_ready.columns)}")
    print(f"sick: train={int(train[CLEAN_TARGET].sum())} ({100 * train[CLEAN_TARGET].mean():.2f} %), "
          f"test={int(test[CLEAN_TARGET].sum())} ({100 * test[CLEAN_TARGET].mean():.2f} %)")
    constant_train = [c for c in train_ready.columns if train_ready[c].nunique() <= 1]
    print(f"Columnas constantes en TRAIN tras transformar: {constant_train or 'ninguna'}")
    # Contamos vectores de atributos repetidos entre particiones: tienen ids distintos y los tratamos como pacientes distintos.
    features = [c for c in data.columns if c != CLEAN_TARGET]
    shared = pd.merge(train[features].drop_duplicates(), test[features].drop_duplicates(), how="inner")
    print(f"Vectores de atributos presentes en train y test a la vez: {len(shared)}")
    print(f"Columnas finales ({train_ready.shape[1]}): {list(train_ready.columns)}")


if __name__ == "__main__":
    run_cleanup()
