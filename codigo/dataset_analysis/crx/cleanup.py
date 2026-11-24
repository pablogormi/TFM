"""Limpieza del dataset Credit Approval (UCI, crx) y preprocesado ajustado en TRAIN.

Ejecución: python cleanup.py (desde cualquier carpeta); parte del CSV crudo y escribe crx_cleaned.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_PATH = SCRIPT_DIR / ".." / ".." / "datasets" / "crx.csv"
CLEANED_PATH = SCRIPT_DIR / "crx_cleaned.csv"
TARGET = "A16"


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# === 15. Limpieza determinista (sin estadísticos aprendidos ni etiqueta) ===
# Mantenemos las filas con ausentes por prudencia: hay cierta evidencia contra MCAR (A11), no concluyente,
# y eliminarlas podría sesgar la muestra; True las elimina.
DROP_MISSING_ROWS = False
CLEAN_BINARY_MAPS = {
    "A1": {"a": 0, "b": 1},
    "A9": {"f": 0, "t": 1},
    "A10": {"f": 0, "t": 1},
    "A12": {"f": 0, "t": 1},
    TARGET: {"-": 0, "+": 1},
}
# Quitamos 't' de A4 porque está documentado pero no aparece nunca y daría una columna one-hot constante.
CLEAN_NOMINAL = {
    "A4": ["u", "y", "l"],
    "A6": ["c", "d", "cc", "i", "j", "k", "m", "r", "q", "w", "x", "e", "aa", "ff"],
    "A7": ["v", "h", "bb", "j", "n", "z", "dd", "ff", "o"],
    "A13": ["g", "p", "s"],
}
CLEAN_CONTINUOUS = ["A2", "A3", "A8", "A11", "A14", "A15"]
CLEAN_INTEGER = ["A11", "A14", "A15"]
# Aplicamos log1p a las no negativas de cola larga (asimetría > 1.4) donde reduce la asimetría.
# La lista se eligió en el EDA sobre las 690 filas y queda fija: no se reajusta en cada partición.
# Dejamos A14 sin log porque su 19.5 % de ceros la volvería bimodal (asimetría -1.26 y 133 atípicos IQR).
LOG_COLUMNS = ["A3", "A8", "A11", "A15"]


def load_raw(path):
    """Leemos crx.csv crudo con los nombres A1..A16 y convertimos `?` a NaN."""
    columns = [f"A{i}" for i in range(1, 17)]
    # Convertimos los `?` a NaN porque crx.names los define como valor ausente.
    return pd.read_csv(path, header=None, names=columns, dtype=str, na_values=["?"], keep_default_na=False)


def clean_deterministic(raw, drop_missing_rows=DROP_MISSING_ROWS):
    """Limpiamos sin estadísticos aprendidos: tipos, niveles, codificación binaria y A5 redundante."""
    df = raw.copy()
    # Quitamos espacios por si una copia futura del CSV los trae; hoy no hay ninguno.
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()
    for col in CLEAN_CONTINUOUS:
        df[col] = pd.to_numeric(df[col], errors="raise")
    # A11, A14 y A15 solo toman valores enteros, así que los guardamos como enteros que admiten NaN.
    for col in CLEAN_INTEGER:
        df[col] = df[col].round().astype("Int64")
    # Anulamos niveles fuera de crx.names para que no aparezcan categorías inventadas (hoy no hay ninguno).
    allowed = {**{c: list(m) for c, m in CLEAN_BINARY_MAPS.items()}, **CLEAN_NOMINAL, "A5": ["g", "p", "gg"]}
    for col, levels in allowed.items():
        invalid = df[col].notna() & ~df[col].isin(levels)
        if invalid.any():
            print(f"  {col}: {int(invalid.sum())} valores fuera de la documentación -> NaN")
            df.loc[invalid, col] = np.nan
    # Eliminamos A5 porque es una recodificación biyectiva de A4 (u<->g, y<->p, l<->gg) y duplicaría la información.
    mapping = pd.crosstab(df["A4"], df["A5"])
    assert ((mapping > 0).sum(axis=1) == 1).all() and ((mapping > 0).sum(axis=0) == 1).all(), "A4/A5 ya no es biyectiva"
    assert (df["A4"].isna() == df["A5"].isna()).all(), "A4 y A5 ya no tienen los mismos ausentes"
    df = df.drop(columns="A5")
    # Codificamos binarias y objetivo como 0/1 con un mapa fijo, sin depender de la frecuencia de los datos.
    for col, mapping_dict in CLEAN_BINARY_MAPS.items():
        df[col] = df[col].map(mapping_dict).astype("Int64")
    if drop_missing_rows:
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        print(f"  Eliminadas {before - len(df)} filas con ausentes")
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


# === 16. Preprocesado ajustado en TRAIN (no se guarda en crx_cleaned.csv) ===
class CrxPreprocessor:
    """Imputamos, agrupamos niveles raros y transformamos; se ajusta en TRAIN y se aplica a cualquier partición.

    Dejamos el escalado a TINTOlib: su transformer se ajusta en fit() con TRAIN y se reutiliza en transform().
    """

    def __init__(self, rare_min_count=0, log_columns=LOG_COLUMNS, clip_quantiles=None, one_hot=True):
        # Con rare_min_count = 0 desactivamos la agrupación; con clip_quantiles = (0.01, 0.99) winsorizamos.
        self.rare_min_count = rare_min_count
        self.log_columns = list(log_columns)
        self.clip_quantiles = clip_quantiles
        self.one_hot = one_hot

    def fit(self, train):
        features = train.drop(columns=TARGET, errors="ignore")
        # Imputamos con la mediana porque las continuas son asimétricas y la media se desplaza por las colas.
        self.medians_ = {c: float(features[c].median()) for c in CLEAN_CONTINUOUS}
        # Imputamos nominales y binarias con la moda de TRAIN, el valor más probable sin usar la etiqueta.
        categorical = [c for c in features.columns if c not in CLEAN_CONTINUOUS]
        self.modes_ = {c: features[c].mode(dropna=True).iloc[0] for c in categorical}
        self.modes_ = {c: (v.item() if hasattr(v, "item") else v) for c, v in self.modes_.items()}
        self.rare_levels_ = {}
        if self.rare_min_count > 0:
            for col, levels in CLEAN_NOMINAL.items():
                counts = features[col].value_counts()
                # Incluimos los niveles ausentes en TRAIN: si no, su columna sería constante en train y activa en test.
                unseen = [level for level in levels if level not in counts.index]
                self.rare_levels_[col] = counts[counts < self.rare_min_count].index.tolist() + unseen
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
        for col, rare in self.rare_levels_.items():
            if rare:
                # Mandamos a 'other' todo lo que no sea un nivel conservado, también valores no documentados.
                kept = [level for level in CLEAN_NOMINAL[col] if level not in rare]
                out[col] = out[col].where(out[col].isna() | out[col].isin(kept), "other")
        for col, (low, high) in self.clip_bounds_.items():
            out[col] = out[col].clip(low, high)
        for col in self.log_columns:
            out[col] = np.log1p(out[col].astype(float))
        if self.one_hot:
            other = {col: (out[col] == "other").astype(float) for col, rare in self.rare_levels_.items() if rare}
            out = one_hot_encode(out)
            # Sustituimos los niveles raros por una única columna 'other', igual en train y test.
            for col, indicator in other.items():
                out = out.drop(columns=[f"{col}_{level}" for level in self.rare_levels_[col] if level in CLEAN_NOMINAL[col]])
                out[f"{col}_other"] = indicator
        # Sin one-hot, las nominales siguen como texto; convertimos a float solo el resto.
        text_columns = [] if self.one_hot else list(CLEAN_NOMINAL)
        for col in [c for c in out.columns if c != TARGET and c not in text_columns]:
            out[col] = out[col].astype(float)
        # Dejamos el objetivo al final porque TINTOlib lo toma de la última columna.
        if TARGET in out.columns:
            out = out[[c for c in out.columns if c != TARGET] + [TARGET]]
        return out

    def fit_transform(self, train):
        return self.fit(train).transform(train)


def run_cleanup():
    section("15. Limpieza determinista")
    raw = load_raw(RAW_PATH)
    print(f"Crudo: {raw.shape}, ausentes totales: {int(raw.isna().sum().sum())}")
    cleaned = clean_deterministic(raw)
    print(f"Limpio: {cleaned.shape}; columnas: {list(cleaned.columns)}")
    print("Cambios: '?' -> NaN; A2, A3, A8 float; A11, A14, A15 enteros; A1/A9/A10/A12/A16 -> 0/1; "
          "A5 eliminada (biyectiva con A4); nominales A4, A6, A7, A13 se mantienen como texto")
    print(f"Ausentes que quedan por columna: {cleaned.isna().sum()[cleaned.isna().sum() > 0].to_dict()}")
    print(f"Objetivo (1 = aprobada): { {int(k): int(v) for k, v in cleaned[TARGET].value_counts().items()} }")
    cleaned.to_csv(CLEANED_PATH, index=False)
    print(f"Guardado {CLEANED_PATH.name} ({len(cleaned)} filas x {cleaned.shape[1]} columnas)")

    section("16. Demostración del preprocesado ajustado en TRAIN")
    # Releemos el CSV limpio para demostrar que el preprocesado funciona desde el fichero entregado.
    data = pd.read_csv(CLEANED_PATH)
    # Estratificamos por A16 para conservar la proporción 44.5/55.5; el 80/20 es solo para la demostración.
    train, test = train_test_split(data, test_size=0.2, stratify=data[TARGET], random_state=SEED)
    preprocessor = CrxPreprocessor().fit(train)
    train_ready = preprocessor.transform(train)
    test_ready = preprocessor.transform(test)
    print(f"Train {train.shape} -> {train_ready.shape}; Test {test.shape} -> {test_ready.shape}")
    print(f"Medianas aprendidas en TRAIN: {preprocessor.medians_}")
    print(f"Modas aprendidas en TRAIN: {preprocessor.modes_}")
    full_medians = {c: float(data[c].median()) for c in CLEAN_CONTINUOUS}
    print(f"(Medianas del dataset completo, NO usadas: {full_medians})")
    print(f"NaN tras transformar: train={int(train_ready.isna().sum().sum())}, test={int(test_ready.isna().sum().sum())}")
    print(f"Columnas iguales en train y test: {list(train_ready.columns) == list(test_ready.columns)}")
    print(f"Proporción de aprobadas: train={train[TARGET].mean():.3f}, test={test[TARGET].mean():.3f}")
    rare_demo = CrxPreprocessor(rare_min_count=5).fit(train)
    print(f"Ejemplo con agrupación de niveles raros (<5 en TRAIN): {rare_demo.rare_levels_}"
          f" -> {rare_demo.transform(test).shape[1]} columnas")


if __name__ == "__main__":
    run_cleanup()
