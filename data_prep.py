# data_prep.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(
    csv_path,
    target_col="Rating",
    min_samples_per_class=50,
    random_state=42
):
    """
    Devuelve:
        X_train_scaled, X_test_scaled, y_train, y_test,
        meta = {
            'class_mapping': dict(idx->label),
            'input_dim': int,
            'num_classes': int,
            'scaler': StandardScaler,
            'feature_cols': [...]
        }
    """

    # Cargar CSV original
    df_raw = pd.read_csv(csv_path)

    # Columnas numéricas que vamos a usar
    num_cols_candidates = [
        "Critic_Score",
        "Critic_Count",
        "User_Score",
        "User_Count",
        "Global_Sales",
        "NA_Sales",
        "EU_Sales",
        "JP_Sales",
        "Other_Sales",
    ]

    # Columnas categóricas
    cat_cols_candidates = [
        "Platform",
        "Genre",
        "Publisher"
    ]

    # Limpiar User_Score ("tbd" y otros strings no numéricos)
    def to_float_or_nan(x):
        try:
            return float(x)
        except:
            return np.nan

    if "User_Score" in df_raw.columns:
        df_raw["User_Score"] = df_raw["User_Score"].apply(to_float_or_nan)

    # Nos quedamos solo con las columnas que queremos
    cols_we_want = num_cols_candidates + cat_cols_candidates + [target_col]
    df = df_raw[cols_we_want].copy()

    # Drop NaN
    df = df.dropna()

    # Filtrar clases con muy pocos datos en la etiqueta
    vc = df[target_col].value_counts()
    valid_classes = vc[vc >= min_samples_per_class].index.tolist()
    df = df[df[target_col].isin(valid_classes)]

    # One-hot de categóricas
    df_encoded = pd.get_dummies(df, columns=cat_cols_candidates, drop_first=True)

    # Features vs Target
    feature_cols = [c for c in df_encoded.columns if c != target_col]
    X_all = df_encoded[feature_cols].values
    y_all = df_encoded[target_col].astype("category").cat.codes.values

    # Mapeo entero -> nombre original de clase
    class_mapping = dict(
        enumerate(df_encoded[target_col].astype("category").cat.categories)
    )

    # Split 80/20 estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        random_state=random_state,
        stratify=y_all
    )

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    input_dim = X_train_scaled.shape[1]
    num_classes = len(np.unique(y_train))

    meta = {
        "class_mapping": class_mapping,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }

    return X_train_scaled, X_test_scaled, y_train, y_test, meta

