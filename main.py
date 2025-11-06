# main.py
# script principal para entrenamie

import os
import glob
import yaml
import torch
import pandas as pd
import kagglehub  # libreria para descargar datasets de kaggle directamente

from data_prep import load_and_prepare_data
from parallel_trainer import run_parallel_training
from evaluate import (
    evaluar_modelo_en_prueba, 
    resumir_sobrevivientes,
    analizar_rendimiento_comparativo,
    generar_reporte_detallado,
    generar_reporte_completo_con_insights,
    crear_tabla_comparativa_final
)


def find_dataset_csv(download_dir):
    """
    busca automaticamente el archivo csv del dataset dentro del directorio descargado
    parametros:
        download_dir: directorio donde kaggle descargo el dataset
    retorna:
        string con la ruta completa al archivo csv encontrado
    """
    # patrones de busqueda para archivos csv tipicos del dataset de videojuegos
    patterns = [
        "Video_Games_Sales*.csv",  # patron especifico para este dataset
        "*.csv"                    # patron generico como respaldo
    ]
    
    # buscar archivos que coincidan con los patrones
    for pattern in patterns:
        candidates = glob.glob(os.path.join(download_dir, pattern))
        if len(candidates) > 0:
            # retornar el primer archivo encontrado
            return candidates[0]

    # si no se encuentra ningun archivo, lanzar excepcion informativa
    raise FileNotFoundError(
        f"No se encontro archivo CSV dentro de {download_dir}. "
        "Revisa manualmente la carpeta descargada."
    )


def main():
    """
    funcion principal que ejecuta todo el pipeline de entrenamiento y evaluacion
    """
    # ======================================================
    # 1. descarga automatica del dataset desde kaggle
    #    kagglehub maneja automaticamente el cache local
    # ======================================================
    print("Descargando dataset de Kaggle (video-game-sales-with-ratings)...")
    print("Esto puede tomar unos minutos en la primera ejecucion...")
    
    # descargar dataset usando kagglehub (crea cache en ~/.cache/kagglehub/)
    dataset_dir = kagglehub.dataset_download(
        "rush4ratio/video-game-sales-with-ratings"
    )
    print("Dataset descargado exitosamente en:", dataset_dir)

    # localizar automaticamente el archivo csv dentro del directorio descargado
    csv_path = find_dataset_csv(dataset_dir)
    print("Archivo CSV ubicado:", csv_path)

    # ======================================================
    # 2. cargar configuraciones de hiperparametros desde archivo externo
    #    esto permite modificar parametros sin cambiar codigo
    # ======================================================
    CONFIG_FILE = "config.yaml"
    print(f"\nCargando configuraciones desde {CONFIG_FILE}...")
    
    # verificar que el archivo de configuracion existe
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(
            f"No se encontro {CONFIG_FILE}. Debe existir en el mismo directorio que main.py"
        )

    # cargar y parsear archivo yaml con configuraciones
    with open(CONFIG_FILE, "r") as f:
        configs_all = yaml.safe_load(f)

    # separar configuraciones por tipo de modelo
    logreg_configs = configs_all["logistic_regression"]
    svm_configs = configs_all["svm"]
    
    print(f"Cargadas {len(logreg_configs)} configuraciones para Regresion Logistica")
    print(f"Cargadas {len(svm_configs)} configuraciones para SVM")

    # ======================================================
    # 3. preparacion y procesamiento de datos
    #    incluye limpieza, codificacion y division train/test
    # ======================================================
    print("\n=== PREPARANDO DATOS ===")
    print("Procesando dataset de videojuegos...")
    
    # cargar y preparar datos con division 80/20
    (
        X_train,        # caracteristicas de entrenamiento (80%)
        X_test,         # caracteristicas de prueba (20%)
        y_train,        # etiquetas de entrenamiento
        y_test,         # etiquetas de prueba
        meta            # metadatos (mapeo de clases, dimensiones, etc.)
    ) = load_and_prepare_data(
        csv_path,
        target_col="Rating",           # columna objetivo: rating del juego
        min_samples_per_class=50,      # minimo de muestras por clase para incluirla
        random_state=42                # semilla para reproducibilidad
    )

    # extraer metadatos importantes
    class_mapping = meta["class_mapping"]  # mapeo indice -> nombre de rating
    input_dim = meta["input_dim"]          # dimensiones de entrada
    num_classes = meta["num_classes"]      # numero de clases diferentes

    print("\n=== INFORMACION DEL DATASET ===")
    print("Forma datos entrenamiento:", X_train.shape)
    print("Forma datos prueba:", X_test.shape)
    print("Mapeo de clases (indice -> rating):", class_mapping)
    print("Dimensiones entrada:", input_dim, "| Numero de clases:", num_classes)
    print("Distribucion aproximada: 80% entrenamiento, 20% prueba")

    # ======================================================
    # 4. Seleccionar device (GPU si está disponible)
    # ======================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nDevice de entrenamiento:", device)

    # ======================================================
    # 5. Entrenar LogReg en paralelo con descarte progresivo
    # ======================================================
    print("\n=== ENTRENANDO LOGISTIC REGRESSION (paralelo multi-config) ===")
    logreg_survivors, logreg_history = run_parallel_training(
        model_type="logreg",
        configs=logreg_configs,
        X_train=X_train,
        y_train=y_train,
        input_dim=input_dim,
        num_classes=num_classes,
        device=device
    )
    print("Sobrevivientes LogReg (top2):",
          [s['cfg']['name'] for s in logreg_survivors])

    # ======================================================
    # 6. Entrenar SVM lineal en paralelo con descarte progresivo
    # ======================================================
    print("\n=== ENTRENANDO SVM (paralelo multi-config) ===")
    svm_survivors, svm_history = run_parallel_training(
        model_type="svm",
        configs=svm_configs,
        X_train=X_train,
        y_train=y_train,
        input_dim=input_dim,
        num_classes=num_classes,
        device=device
    )
    print("Sobrevivientes SVM (top2):",
          [s['cfg']['name'] for s in svm_survivors])

    # ======================================================
    # 7. Contexto de las clasificaciones ESRB
    # ======================================================
    print("\n" + "="*70)
    print("CONTEXTO: SIGNIFICADO DE LOS RATINGS ESRB")
    print("="*70)
    print("E (Everyone): Contenido apropiado para todas las edades")
    print("E10+ (Everyone 10+): Contenido apropiado para mayores de 10 años")
    print("T (Teen): Contenido apropiado para mayores de 13 años")
    print("M (Mature 17+): Contenido apropiado para mayores de 17 años")
    print("="*70)

    # ======================================================
    # 8. Evaluación final de las mejores configuraciones en TEST
    #    (esto es lo que luego pones en el informe)
    # ======================================================
    print("\n=== RESULTADOS EN TEST: LOGREG TOP2 ===")
    resultados_logreg = []
    for s in logreg_survivors:
        result = evaluar_modelo_en_prueba(
            s,
            X_test,
            y_test,
            class_mapping,
            device=device
        )
        resultados_logreg.append(result)
        print("\n-------------------------------------------------")
        print("Configuracion:", result["nombre_configuracion"])
        print("Hiperparametros:", result["hiperparametros"])
        print("Epocas entrenadas:", result["epocas_entrenadas"])
        print("Precision final entrenamiento:", result["precision_final_entrenamiento"])
        print("Perdida final entrenamiento:", result["perdida_final_entrenamiento"])
        print("Precision en prueba:", f"{result['precision_prueba']:.4f} ({result['precision_prueba']*100:.2f}%)")
        print("F1-Score macro:", f"{result['f1_macro']:.4f}")
        print("Precision macro:", f"{result['precision_macro']:.4f}")
        print("Recall macro:", f"{result['recall_macro']:.4f}")
        print("Reporte de clasificacion:\n", result["reporte_clasificacion"])
        print("Matriz de confusion:\n", result["matriz_confusion"])

    print("\n=== RESULTADOS EN TEST: SVM TOP2 ===")
    resultados_svm = []
    for s in svm_survivors:
        result = evaluar_modelo_en_prueba(
            s,
            X_test,
            y_test,
            class_mapping,
            device=device
        )
        resultados_svm.append(result)
        print("\n-------------------------------------------------")
        print("Configuracion:", result["nombre_configuracion"])
        print("Hiperparametros:", result["hiperparametros"])
        print("Epocas entrenadas:", result["epocas_entrenadas"])
        print("Precision final entrenamiento:", result["precision_final_entrenamiento"])
        print("Perdida final entrenamiento:", result["perdida_final_entrenamiento"])
        print("Precision en prueba:", f"{result['precision_prueba']:.4f} ({result['precision_prueba']*100:.2f}%)")
        print("F1-Score macro:", f"{result['f1_macro']:.4f}")
        print("Precision macro:", f"{result['precision_macro']:.4f}")
        print("Recall macro:", f"{result['recall_macro']:.4f}")
        print("Reporte de clasificacion:\n", result["reporte_clasificacion"])
        print("Matriz de confusion:\n", result["matriz_confusion"])

    # ======================================================
    # 9. Tabla-resumen de los dos mejores por técnica
    #    (no es análisis, es solo resumen objetivo)
    # ======================================================
    logreg_summary = resumir_sobrevivientes(logreg_survivors)
    svm_summary = resumir_sobrevivientes(svm_survivors)

    print("\n=== RESUMEN TOP LOGREG (para copiar en informe) ===")
    print(pd.DataFrame(logreg_summary))

    print("\n=== RESUMEN TOP SVM (para copiar en informe) ===")
    print(pd.DataFrame(svm_summary))

    # ======================================================
    # 10. Analisis comparativo detallado y reportes
    # ======================================================
    print("\n" + "="*80)
    print("ANALISIS COMPARATIVO DETALLADO")
    print("="*80)
    
    # generar analisis comparativo
    analisis_comparativo = analizar_rendimiento_comparativo(resultados_logreg, resultados_svm)
    
    # mostrar resumen del analisis
    if analisis_comparativo["comparacion_general"]:
        comp = analisis_comparativo["comparacion_general"]
        print(f"\nMODELO SUPERIOR: {comp['modelo_superior']}")
        print(f"Diferencia de precision: {comp['diferencia_precision']:.4f}")
        print(f"Ventaja de Regresion Logistica: {comp['ventaja_logreg']:.4f}")
        print(f"Mejor configuracion global: {comp['mejor_configuracion_global']}")
    
    # mostrar recomendaciones
    if analisis_comparativo["recomendaciones"]:
        print("\nRECOMENDACIONES:")
        for i, rec in enumerate(analisis_comparativo["recomendaciones"], 1):
            print(f"{i}. {rec}")
    
    # generar reportes detallados
    print(generar_reporte_detallado(resultados_logreg, "Regresion Logistica"))
    print(generar_reporte_detallado(resultados_svm, "SVM"))
    
    # ======================================================
    # 11. Analisis profundo con insights adicionales
    # ======================================================
    print(generar_reporte_completo_con_insights(
        resultados_logreg, resultados_svm, 
        logreg_survivors, svm_survivors,
        class_mapping, X_train, X_test
    ))
    
    # ======================================================
    # 12. Tabla comparativa final y recomendaciones
    # ======================================================
    print(crear_tabla_comparativa_final(resultados_logreg, resultados_svm))
    
    


if __name__ == "__main__":
    main()

