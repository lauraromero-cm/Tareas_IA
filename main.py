# main.py

import os
import glob
import yaml
import torch
import pandas as pd
import kagglehub  # <-- usamos kagglehub directamente

from data_prep import load_and_prepare_data
from parallel_trainer import run_parallel_training
from evaluate import (
    eval_model_on_test, 
    summarize_survivors,
    formatear_tabla_resumen,
    analizar_rendimiento_comparativo,
    generar_reporte_detallado,
    generar_reporte_completo_con_insights,
    crear_tabla_comparativa_final
)


def find_dataset_csv(download_dir):
    """
    Busca dentro de la carpeta descargada el CSV del dataset
    de ventas de videojuegos. Devuelve la primera coincidencia.
    """
    # buscamos archivos .csv típicos del dataset
    patterns = [
        "Video_Games_Sales*.csv",
        "*.csv"
    ]
    for pattern in patterns:
        candidates = glob.glob(os.path.join(download_dir, pattern))
        if len(candidates) > 0:
            # tomamos el primero
            return candidates[0]

    raise FileNotFoundError(
        f"No se encontró CSV dentro de {download_dir}. "
        "Revisa manualmente la carpeta descargada."
    )


def main():
    # ======================================================
    # 1. Descargar dataset con kagglehub
    #    Esto crea/usa un caché local (~/.cache/kagglehub/...)
    # ======================================================
    print("Descargando dataset de Kaggle (video-game-sales-with-ratings)...")
    dataset_dir = kagglehub.dataset_download(
        "rush4ratio/video-game-sales-with-ratings"
    )
    print("Dataset descargado en:", dataset_dir)

    # Ubicar CSV automáticamente
    csv_path = find_dataset_csv(dataset_dir)
    print("Usando CSV:", csv_path)

    # ======================================================
    # 2. Cargar hiperparámetros externos desde config.yaml
    # ======================================================
    CONFIG_FILE = "config.yaml"
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(
            f"No se encontró {CONFIG_FILE}. Debe existir en el mismo directorio."
        )

    with open(CONFIG_FILE, "r") as f:
        configs_all = yaml.safe_load(f)

    logreg_configs = configs_all["logistic_regression"]
    svm_configs = configs_all["svm"]

    # ======================================================
    # 3. Preparar datos
    # ======================================================
    (
        X_train,
        X_test,
        y_train,
        y_test,
        meta
    ) = load_and_prepare_data(
        csv_path,
        target_col="Rating",
        min_samples_per_class=50,
        random_state=42
    )

    class_mapping = meta["class_mapping"]
    input_dim = meta["input_dim"]
    num_classes = meta["num_classes"]

    print("\n=== INFO DATA ===")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Clases (id -> etiqueta Rating):", class_mapping)
    print("CONTEXTO DE RATINGS:")
    print("  • E: Everyone (Apto para todas las edades)")
    print("  • E10+: Everyone 10+ (Apto para mayores de 10 años)")
    print("  • T: Teen (Apto para adolescentes 13+)")
    print("  • M: Mature (Apto para adultos 17+)")
    print("input_dim:", input_dim, "| num_classes:", num_classes)

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
    # 7. Evaluación final de las mejores configuraciones en TEST
    # ======================================================
    print("\n=== EVALUACION EN CONJUNTO DE PRUEBA ===")
    
    # Evaluar todas las configuraciones supervivientes
    resultados_logreg = []
    for s in logreg_survivors:
        result = eval_model_on_test(s, X_test, y_test, class_mapping, device=device)
        resultados_logreg.append(result)
    
    resultados_svm = []
    for s in svm_survivors:
        result = eval_model_on_test(s, X_test, y_test, class_mapping, device=device)
        resultados_svm.append(result)

    # ======================================================
    # 8. RESÚMENES Y ANÁLISIS DETALLADO
    # ======================================================
    
    # Resúmenes formateados
    logreg_summary = summarize_survivors(logreg_survivors)
    svm_summary = summarize_survivors(svm_survivors)
    
    print(formatear_tabla_resumen(logreg_summary, "REGRESION LOGISTICA"))
    print(formatear_tabla_resumen(svm_summary, "SVM"))
    
    # Reportes detallados por modelo
    print(generar_reporte_detallado(resultados_logreg, "Regresion Logistica"))
    print(generar_reporte_detallado(resultados_svm, "SVM"))
    
    # ======================================================
    # 9. RESULTADOS DETALLADOS 
    # ======================================================
    print("\n" + "="*100)
    print("RESULTADOS DETALLADOS DE LAS MEJORES CONFIGURACIONES")
    print("="*100)
    
    print("\n--- REGRESION LOGISTICA: MEJORES 2 CONFIGURACIONES ---")
    for i, result in enumerate(resultados_logreg, 1):
        print(f"\nCONFIGURACION {i}: {result['config_name']}")
        print(f"Precision en Prueba: {result['test_accuracy']:.4f} ({result['test_accuracy']*100:.2f}%)")
        print(f"Epocas entrenadas: {result['train_epochs']}")
        print(f"Precision final entrenamiento: {result['final_train_acc']:.4f}")
        print(f"Perdida final entrenamiento: {result['final_train_loss']:.4f}")
        print(f"Hiperparametros: {result['hyperparams']}")
        print("Reporte de clasificacion:")
        print(result['classification_report'])
        print("Matriz de confusion:")
        print(result['confusion_matrix'])
    
    print("\n--- SVM: MEJORES 2 CONFIGURACIONES ---")
    for i, result in enumerate(resultados_svm, 1):
        print(f"\nCONFIGURACION {i}: {result['config_name']}")
        print(f"Precision en Prueba: {result['test_accuracy']:.4f} ({result['test_accuracy']*100:.2f}%)")
        print(f"Epocas entrenadas: {result['train_epochs']}")
        print(f"Precision final entrenamiento: {result['final_train_acc']:.4f}")
        print(f"Perdida final entrenamiento: {result['final_train_loss']:.4f}")
        print(f"Hiperparametros: {result['hyperparams']}")
        print("Reporte de clasificacion:")
        print(result['classification_report'])
        print("Matriz de confusion:")
        print(result['confusion_matrix'])
    
    # ======================================================
    # 10. CONCLUSIONES FINALES
    # ======================================================
    
    # Análisis comparativo detallado
    analisis_comparativo = analizar_rendimiento_comparativo(resultados_logreg, resultados_svm)
    
    # Reporte integral con insights profundos
    print(generar_reporte_completo_con_insights(
        resultados_logreg, resultados_svm,
        logreg_survivors, svm_survivors,
        class_mapping, X_train, X_test
    ))
    
    # Tabla comparativa final ejecutiva
    print(crear_tabla_comparativa_final(resultados_logreg, resultados_svm))
    
    # ======================================================
    # 11. CONCLUSIONES FINALES 
    # ======================================================
    if resultados_logreg and resultados_svm:
        mejor_logreg = max(resultados_logreg, key=lambda x: x["test_accuracy"])
        mejor_svm = max(resultados_svm, key=lambda x: x["test_accuracy"])
        
        print("\n" + "="*100)
        print("CONCLUSIONES FINALES AUTOMATIZADAS")
        print("="*100)
        
        print(f"\nMEJOR MODELO GENERAL:")
        if mejor_logreg["test_accuracy"] > mejor_svm["test_accuracy"]:
            ganador = mejor_logreg
            tipo_ganador = "Regresion Logistica"
            diferencia = mejor_logreg["test_accuracy"] - mejor_svm["test_accuracy"]
        else:
            ganador = mejor_svm
            tipo_ganador = "SVM"
            diferencia = mejor_svm["test_accuracy"] - mejor_logreg["test_accuracy"]
        
        print(f"- Modelo ganador: {tipo_ganador}")
        print(f"- Configuracion: {ganador['config_name']}")
        print(f"- Precision maxima: {ganador['test_accuracy']:.4f} ({ganador['test_accuracy']*100:.2f}%)")
        print(f"- Ventaja sobre competidor: {diferencia:.4f} ({diferencia*100:.2f}%)")
        
        print(f"\nFACTORES CLAVE DEL EXITO:")
        hp = ganador['hyperparams']
        print(f"- Tasa de aprendizaje: {hp['lr']}")
        print(f"- Tamaño de lote: {hp['batch_size']}")
        print(f"- Regularizacion: {hp.get('weight_decay', 'N/A')}")
        if 'C' in hp:
            print(f"- Parametro C (SVM): {hp['C']}")
            print(f"- Margen (SVM): {hp.get('margin', 'N/A')}")
        
        if diferencia < 0.02:
            print(f"\nRECOMENDACION: La diferencia es minima ({diferencia*100:.2f}%).")
            print("Se sugiere elegir Regresion Logistica por simplicidad computacional.")
        else:
            print(f"\nRECOMENDACION: La diferencia es significativa ({diferencia*100:.2f}%).")
            print(f"Se recomienda usar {tipo_ganador} para maximizar precision.")
    
   


if __name__ == "__main__":
    main()

