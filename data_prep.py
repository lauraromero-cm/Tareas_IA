# data_prep.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde





def load_and_prepare_data(
    csv_path,
    target_col="Rating",
    min_samples_per_class=50,
    apply_augmentation=True,
    target_augmentation_size=10000,
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

    # aplicar data augmentation silenciosamente si se solicita
    if apply_augmentation:
        X_train_scaled, y_train = aplicar_data_augmentation(
            X_train_scaled, 
            y_train, 
            target_size=target_augmentation_size,
            random_state=random_state
        )

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

def aplicar_data_augmentation(X_train, y_train, target_size=10000, random_state=42):
    np.random.seed(random_state)
    
    current_size = len(X_train)
    samples_needed = target_size - current_size
    
    
    if samples_needed <= 0:
        return X_train, y_train
    
    
    X_augmented_list = [X_train]
    y_augmented_list = [y_train]
    
    
    clases_unicas, counts = np.unique(y_train, return_counts=True)
    
    
    samples_per_class = {}
    for clase, count in zip(clases_unicas, counts):
        proporcion = count / current_size
        samples_clase = int(samples_needed * proporcion)
        samples_per_class[clase] = max(samples_clase, 1)  
    
    
    for clase in clases_unicas:
        num_to_generate = samples_per_class[clase]
        
        
        mask_clase = y_train == clase
        X_clase = X_train[mask_clase]
        
        if len(X_clase) < 2:
            # si hay muy pocas muestras, usar ruido gaussiano simple
            X_synthetic = _generar_con_ruido_gaussiano(X_clase, num_to_generate)
        else:
            # usar SMOTE-like approach para generar muestras sinteticas
            X_synthetic = _generar_con_smote_like(X_clase, num_to_generate)
        
        # agregar las muestras sinteticas
        if len(X_synthetic) > 0:
            X_augmented_list.append(X_synthetic)
            y_augmented_list.append(np.full(len(X_synthetic), clase))
    
    
    X_augmented = np.vstack(X_augmented_list)
    y_augmented = np.hstack(y_augmented_list)
    
    
    indices = np.random.permutation(len(X_augmented))
    X_augmented = X_augmented[indices]
    y_augmented = y_augmented[indices]
    
    
    return X_augmented, y_augmented


def _generar_con_smote_like(X_clase, num_to_generate):
    
    if len(X_clase) < 2 or num_to_generate <= 0:
        return np.array([]).reshape(0, X_clase.shape[1])
    
    
    k = min(5, len(X_clase) - 1)  
    if k <= 0:
        return _generar_con_ruido_gaussiano(X_clase, num_to_generate)
    
    try:
        nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        nn.fit(X_clase)
    except:
        
        return _generar_con_ruido_gaussiano(X_clase, num_to_generate)
    
    muestras_sinteticas = []
    
    for _ in range(num_to_generate):
       
        idx = np.random.randint(0, len(X_clase))
        muestra_base = X_clase[idx]
        
        
        try:
            distancias, indices = nn.kneighbors([muestra_base])
            vecino_idx = np.random.choice(indices[0])
            vecino = X_clase[vecino_idx]
            
            
            alpha = np.random.random()  
            muestra_sintetica = muestra_base + alpha * (vecino - muestra_base)
            
            
            ruido = np.random.normal(0, 0.01, muestra_sintetica.shape)
            muestra_sintetica += ruido
            
            muestras_sinteticas.append(muestra_sintetica)
            
        except:
            
            ruido = np.random.normal(0, 0.05, muestra_base.shape)
            muestra_sintetica = muestra_base + ruido
            muestras_sinteticas.append(muestra_sintetica)
    
    return np.array(muestras_sinteticas)


def _generar_con_ruido_gaussiano(X_clase, num_to_generate):
    if len(X_clase) == 0 or num_to_generate <= 0:
        return np.array([]).reshape(0, X_clase.shape[1] if len(X_clase) > 0 else 0)
    
    muestras_sinteticas = []
    
    
    media = np.mean(X_clase, axis=0)
    std = np.std(X_clase, axis=0)
    std = np.where(std == 0, 0.01, std)  
    
    for _ in range(num_to_generate):
        
        idx = np.random.randint(0, len(X_clase))
        muestra_base = X_clase[idx].copy()
        
       
        ruido = np.random.normal(0, 0.1 * std)
        muestra_sintetica = muestra_base + ruido
        
        muestras_sinteticas.append(muestra_sintetica)
    
    return np.array(muestras_sinteticas)