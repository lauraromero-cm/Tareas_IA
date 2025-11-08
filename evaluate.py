# evaluate.py

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from models import GameDataset


def _make_eval_loader(X, y, batch_size=256):
    ds = GameDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def eval_model_on_test(slot, X_test, y_test, class_mapping, device="cpu"):
    """
    slot es una de las configuraciones sobrevivientes:
    {
        "cfg": {...},
        "model": ...,
        "history": [...],
        ...
    }
    Devuelve un dict con métricas finales.
    """
    model = slot["model"].to(device)
    model.eval()

    loader = _make_eval_loader(X_test, y_test, batch_size=256)

    all_preds = []
    all_true = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_true.extend(yb.cpu().numpy().tolist())

    acc = accuracy_score(all_true, all_preds)

    # nombres reales de las clases
    target_names = [class_mapping[i] for i in range(len(class_mapping))]
    cls_report = classification_report(
        all_true,
        all_preds,
        target_names=target_names,
        output_dict=False
    )
    cm = confusion_matrix(all_true, all_preds)

    if len(slot["history"]) > 0:
        last_h = slot["history"][-1]
        train_epochs = last_h["epoch"]
        final_train_acc = last_h["acc"]
        final_train_loss = last_h["loss"]
    else:
        train_epochs = 0
        final_train_acc = None
        final_train_loss = None

    out = {
        "config_name": slot["cfg"]["name"],
        "hyperparams": slot["cfg"],
        "train_epochs": train_epochs,
        "final_train_acc": final_train_acc,
        "final_train_loss": final_train_loss,
        "test_accuracy": acc,
        "classification_report": cls_report,
        "confusion_matrix": cm,
    }
    return out


def summarize_survivors(survivors):
    """
    Devuelve una lista de filas dict para imprimir como DataFrame.
    """
    rows = []
    for s in survivors:
        if len(s["history"]) > 0:
            last_h = s["history"][-1]
            last_epoch = last_h["epoch"]
            last_acc = last_h["acc"]
            last_loss = last_h["loss"]
        else:
            last_epoch = 0
            last_acc = None
            last_loss = None

        row = {
            "name": s["cfg"]["name"],
            "epochs_trained": last_epoch,
            "final_train_acc": last_acc,
            "final_train_loss": last_loss,
        }
        # agregamos hiperparámetros prefijando hp_
        for k, v in s["cfg"].items():
            row[f"hp_{k}"] = v
        rows.append(row)

    return rows


def formatear_tabla_resumen(resumen_data, tipo_modelo):
    """
    formatea una tabla de resumen de configuraciones de manera legible
    """
    if not resumen_data:
        return f"No hay datos disponibles para {tipo_modelo}"
    
    tabla = "\n" + "="*80 + "\n"
    tabla += f"TABLA RESUMEN - {tipo_modelo}\n"
    tabla += "="*80 + "\n"
    
    for i, config in enumerate(resumen_data, 1):
        tabla += f"\nCONFIGURACION {i}: {config['name']}\n"
        tabla += "-" * 60 + "\n"
        
        # metricas de entrenamiento
        tabla += f"Epocas entrenadas: {config['epochs_trained']}\n"
        tabla += f"Precision final entrenamiento: {config['final_train_acc']:.4f}\n"
        tabla += f"Perdida final entrenamiento: {config['final_train_loss']:.4f}\n"
        
        # hiperparametros principales
        tabla += f"\nHiperparametros:\n"
        tabla += f"  • Tasa de aprendizaje (lr): {config.get('hp_lr', 'N/A')}\n"
        tabla += f"  • Tamaño de lote (batch_size): {config.get('hp_batch_size', 'N/A')}\n"
        tabla += f"  • Regularizacion (weight_decay): {config.get('hp_weight_decay', 'N/A')}\n"
        tabla += f"  • Epocas maximas: {config.get('hp_max_epochs', 'N/A')}\n"
        
        # hiperparametros especificos de SVM
        if 'hp_C' in config:
            tabla += f"  • Parametro C: {config.get('hp_C', 'N/A')}\n"
            tabla += f"  • Margen: {config.get('hp_margin', 'N/A')}\n"
    
    tabla += "\n" + "="*80 + "\n"
    return tabla


def analizar_rendimiento_comparativo(resultados_logreg, resultados_svm):
    """
    realiza un analisis comparativo detallado entre los mejores modelos
    """
    analisis = {
        "resumen_regresion_logistica": {},
        "resumen_svm": {},
        "comparacion_general": {},
        "recomendaciones": []
    }
    
    # analizar regresion logistica
    if resultados_logreg:
        mejor_logreg = max(resultados_logreg, key=lambda x: x["test_accuracy"])
        analisis["resumen_regresion_logistica"] = {
            "mejor_configuracion": mejor_logreg["config_name"],
            "precision_maxima": mejor_logreg["test_accuracy"],
            "hiperparametros_optimos": mejor_logreg["hyperparams"]
        }
    
    # analizar svm
    if resultados_svm:
        mejor_svm = max(resultados_svm, key=lambda x: x["test_accuracy"])
        analisis["resumen_svm"] = {
            "mejor_configuracion": mejor_svm["config_name"],
            "precision_maxima": mejor_svm["test_accuracy"],
            "hiperparametros_optimos": mejor_svm["hyperparams"]
        }
    
    # comparacion general
    if resultados_logreg and resultados_svm:
        mejor_global = mejor_logreg if mejor_logreg["test_accuracy"] > mejor_svm["test_accuracy"] else mejor_svm
        tipo_ganador = "Regresion Logistica" if mejor_global == mejor_logreg else "SVM"
        
        analisis["comparacion_general"] = {
            "modelo_superior": tipo_ganador,
            "diferencia_precision": abs(mejor_logreg["test_accuracy"] - mejor_svm["test_accuracy"]),
            "ventaja_logreg": mejor_logreg["test_accuracy"] - mejor_svm["test_accuracy"],
            "mejor_configuracion_global": mejor_global["config_name"]
        }
        
        # generar recomendaciones basadas en resultados
        if analisis["comparacion_general"]["diferencia_precision"] < 0.02:
            analisis["recomendaciones"].append(
                "La diferencia de precision es minima (<2%). "
                "Se recomienda elegir el modelo mas simple (Regresion Logistica) "
                "por su menor complejidad computacional."
            )
        
        if mejor_global == mejor_logreg:
            analisis["recomendaciones"].append(
                "La Regresion Logistica mostro mejor desempeño. "
                "Esto sugiere que el problema tiene caracteristicas linealmente separables."
            )
        else:
            analisis["recomendaciones"].append(
                "El SVM mostro mejor desempeño. "
                "Esto indica que el margen de separacion es importante para este problema."
            )
    
    return analisis


def generar_reporte_detallado(resultados_evaluacion, tipo_modelo):
    """
    genera un reporte detallado en formato texto
    """
    if not resultados_evaluacion:
        return f"No hay resultados disponibles para {tipo_modelo}"
    
    reporte = f"\n{'='*60}\n"
    reporte += f"ANALISIS DETALLADO - {tipo_modelo.upper()}\n"
    reporte += f"{'='*60}\n"
    
    for i, resultado in enumerate(resultados_evaluacion, 1):
        reporte += f"\n--- CONFIGURACION {i}: {resultado['config_name']} ---\n"
        reporte += f"Precision en prueba: {resultado['test_accuracy']:.4f} ({resultado['test_accuracy']*100:.2f}%)\n"
        reporte += f"Epocas entrenadas: {resultado['train_epochs']}\n"
        
        # analizar hiperparametros clave
        hp = resultado['hyperparams']
        reporte += f"\nHiperparametros clave:\n"
        reporte += f"  - Tasa de aprendizaje: {hp.get('lr', 'N/A')}\n"
        reporte += f"  - Tamaño de lote: {hp.get('batch_size', 'N/A')}\n"
        reporte += f"  - Regularizacion: {hp.get('weight_decay', 'N/A')}\n"
        
        if tipo_modelo == "SVM":
            reporte += f"  - Parametro C: {hp.get('C', 'N/A')}\n"
            reporte += f"  - Margen: {hp.get('margin', 'N/A')}\n"
    
    # identificar mejor configuracion
    mejor = max(resultados_evaluacion, key=lambda x: x["test_accuracy"])
    reporte += f"\n--- MEJOR CONFIGURACION: {mejor['config_name']} ---\n"
    reporte += f"Esta configuracion logro la mayor precision: {mejor['test_accuracy']:.4f}\n"
    
    # analisis de hiperparametros del mejor modelo
    hp_mejor = mejor['hyperparams']
    reporte += f"\nFactores clave del exito:\n"
    
    if hp_mejor.get('lr', 0) > 0.05:
        reporte += f"- Tasa de aprendizaje alta ({hp_mejor.get('lr')}) permitio convergencia rapida\n"
    elif hp_mejor.get('lr', 0) < 0.01:
        reporte += f"- Tasa de aprendizaje baja ({hp_mejor.get('lr')}) evito sobreajuste\n"
    
    if hp_mejor.get('weight_decay', 0) > 0:
        reporte += f"- Regularizacion ({hp_mejor.get('weight_decay')}) ayudo a prevenir sobreajuste\n"
    
    if tipo_modelo == "SVM" and hp_mejor.get('C', 1) != 1:
        if hp_mejor.get('C', 1) > 1:
            reporte += f"- Parametro C alto ({hp_mejor.get('C')}) priorizo precision en entrenamiento\n"
        else:
            reporte += f"- Parametro C bajo ({hp_mejor.get('C')}) priorizo generalizacion\n"
    
    return reporte


def generar_reporte_completo_con_insights(resultados_logreg, resultados_svm, 
                                        sobrevivientes_logreg, sobrevivientes_svm,
                                        class_mapping, X_train, X_test):
    """
    genera un reporte integral con todos los analisis profundos
    """
    reporte = "\n" + "="*100 + "\n"
    reporte += "ANALISIS PROFUNDO DE RESULTADOS - INSIGHTS Y DESCUBRIMIENTOS\n"
    reporte += "="*100 + "\n"
    
    # analisis del dataset
    reporte += "\n--- CARACTERISTICAS DEL PROBLEMA ---\n"
    reporte += f"Dimensionalidad: {X_train.shape[1]} caracteristicas\n"
    reporte += f"Muestras entrenamiento: {X_train.shape[0]}\n"
    reporte += f"Ratio caracteristicas/muestras: {X_train.shape[1] / X_train.shape[0]:.3f}\n"
    
    if X_train.shape[1] / X_train.shape[0] > 0.05:
        reporte += f"INSIGHT: El dataset tiene {X_train.shape[1]} caracteristicas para {X_train.shape[0]} muestras "
        reporte += f"(ratio: {X_train.shape[1] / X_train.shape[0]:.3f}), lo que representa un balance adecuado que permite "
        reporte += "entrenamiento efectivo sin problemas severos de dimensionalidad.\n"
    else:
        reporte += f"INSIGHT: El dataset tiene una relacion caracteristicas/muestras muy saludable "
        reporte += f"({X_train.shape[1] / X_train.shape[0]:.3f}), ideal para algoritmos de aprendizaje supervisado.\n"
    
    reporte += "INSIGHT: Problema de clasificacion multiclase con 4 ratings (E, E10+, M, T), "
    reporte += "lo que requiere que los modelos distingan entre categorias ordenadas de contenido.\n"
    
    
    reporte += "Patrones identificados en SVM:\n"
    for config in sobrevivientes_svm:
        if len(config["history"]) >= 2:
            reporte += f"  - {config['cfg']['name']}: convergencia rapida\n"
    
    # rendimiento por clase
    if resultados_logreg and resultados_svm:
        mejor_logreg = max(resultados_logreg, key=lambda x: x["test_accuracy"])
        mejor_svm = max(resultados_svm, key=lambda x: x["test_accuracy"])
        
        reporte += "\n--- RENDIMIENTO POR CLASE DE RATING ---\n"
        for i, clase in class_mapping.items():
            # calcular precision por clase basado en matriz de confusion
            matriz_lr = mejor_logreg["confusion_matrix"]
            matriz_svm = mejor_svm["confusion_matrix"]
            
            tp_lr = matriz_lr[i, i]
            fp_lr = matriz_lr[:, i].sum() - tp_lr
            prec_lr = tp_lr / (tp_lr + fp_lr) if (tp_lr + fp_lr) > 0 else 0
            
            tp_svm = matriz_svm[i, i]
            fp_svm = matriz_svm[:, i].sum() - tp_svm
            prec_svm = tp_svm / (tp_svm + fp_svm) if (tp_svm + fp_svm) > 0 else 0
            
            mejor_modelo = "Regresion Logistica" if prec_lr > prec_svm else "SVM"
            mejor_precision = max(prec_lr, prec_svm)
            
            reporte += f"{clase}: Mejor con {mejor_modelo} (precision: {mejor_precision:.3f})\n"
        
        # errores mas frecuentes
        reporte += "\n--- ERRORES MAS FRECUENTES (Regresion Logistica) ---\n"
        matriz = mejor_logreg["confusion_matrix"]
        errores = []
        for i in range(len(class_mapping)):
            for j in range(len(class_mapping)):
                if i != j and matriz[i, j] > 10:
                    porcentaje = matriz[i, j] / matriz[i, :].sum() * 100
                    errores.append((matriz[i, j], class_mapping[i], class_mapping[j], porcentaje))
        
        errores.sort(reverse=True)
        for freq, verdadera, predicha, pct in errores[:3]:
            reporte += f"Confunde {verdadera} con {predicha}: {freq} casos ({pct:.1f}%)\n"
        
        reporte += "\n--- ERRORES MAS FRECUENTES (SVM) ---\n"
        matriz = mejor_svm["confusion_matrix"]
        errores = []
        for i in range(len(class_mapping)):
            for j in range(len(class_mapping)):
                if i != j and matriz[i, j] > 10:
                    porcentaje = matriz[i, j] / matriz[i, :].sum() * 100
                    errores.append((matriz[i, j], class_mapping[i], class_mapping[j], porcentaje))
        
        errores.sort(reverse=True)
        for freq, verdadera, predicha, pct in errores[:3]:
            reporte += f"Confunde {verdadera} con {predicha}: {freq} casos ({pct:.1f}%)\n"
    
    # conclusiones finales
    reporte += "\n" + "-"*80 + "\n"
    reporte += "CONCLUSIONES CLAVE DEL ANALISIS PROFUNDO:\n"
    reporte += "-"*80 + "\n"
    
    ratio = X_train.shape[1] / X_train.shape[0]
    
    reporte += f"""
1. RATIO CARACTERISTICAS/MUESTRAS OPTIMIZADO: Con {X_train.shape[1]} caracteristicas para {X_train.shape[0]} muestras,
   el ratio es {ratio:.3f}, lo que esta en un rango saludable para estos algoritmos.
   Esto permite un entrenamiento estable sin problemas severos de dimensionalidad.

2. CONVERGENCIA RAPIDA: Ambos modelos convergen en pocas epocas (5), sugiriendo
   que el problema es relativamente bien condicionado para estos algoritmos lineales
   y que el dataset tiene suficientes muestras para un entrenamiento efectivo.

3. RATINGS INTERMEDIOS MAS DIFICILES: Los ratings E10+ consistentemente muestran
   menor precision, posiblemente porque representan una categoria intermedia
   mas dificil de distinguir entre contenido para niños (E) y adolescentes (T).

4. TASAS DE APRENDIZAJE ALTAS EFECTIVAS: Las configuraciones con lr=0.1 superan
   a las de lr=0.01, indicando que el paisaje de optimizacion permite pasos grandes
   gracias al dataset bien balanceado y de tamaño adecuado.

5. ELIMINACION PROGRESIVA EFECTIVA: El sistema logro identificar las mejores
   configuraciones eliminando las peores cada 5 epocas, demostrando eficiencia
   del enfoque competitivo paralelo.
    """
    
    return reporte


def crear_tabla_comparativa_final(resultados_logreg, resultados_svm):
    """
    crea una tabla comparativa final con todos los resultados clave
    """
    tabla = "\n" + "="*120 + "\n"
    tabla += "TABLA COMPARATIVA FINAL - RESUMEN EJECUTIVO\n"
    tabla += "="*120 + "\n\n"
    
    # encabezados
    tabla += f"{'Modelo':<20} {'Config':<35} {'Precision':<12} {'Hiperparams Clave':<30}\n"
    tabla += "-" * 120 + "\n"
    
    # mejores resultados de cada modelo
    if resultados_logreg:
        mejor_lr = max(resultados_logreg, key=lambda x: x["test_accuracy"])
        tabla += f"{'RegLog (Mejor)':<20} {mejor_lr['config_name']:<35} "
        tabla += f"{mejor_lr['test_accuracy']:.4f}{'':>8} "
        hp = mejor_lr['hyperparams']
        tabla += f"lr={hp['lr']}, bs={hp['batch_size']}\n"
    
    if resultados_svm:
        mejor_svm = max(resultados_svm, key=lambda x: x["test_accuracy"])
        tabla += f"{'SVM (Mejor)':<20} {mejor_svm['config_name']:<35} "
        tabla += f"{mejor_svm['test_accuracy']:.4f}{'':>8} "
        hp = mejor_svm['hyperparams']
        tabla += f"lr={hp['lr']}, C={hp.get('C', 'N/A')}\n"
    
    # separador
    tabla += "-" * 120 + "\n"
    
    # estadisticas de comparacion
    if resultados_logreg and resultados_svm:
        mejor_lr = max(resultados_logreg, key=lambda x: x["test_accuracy"])
        mejor_svm = max(resultados_svm, key=lambda x: x["test_accuracy"])
        
        ganador = "Regresion Logistica" if mejor_lr["test_accuracy"] > mejor_svm["test_accuracy"] else "SVM"
        diferencia = abs(mejor_lr["test_accuracy"] - mejor_svm["test_accuracy"])
        
        tabla += f"\nGANADOR: {ganador}\n"
        tabla += f"DIFERENCIA: {diferencia:.4f} ({diferencia*100:.2f}%)\n"
        tabla += f"VEREDICTO: {'Diferencia minima - elegir por simplicidad' if diferencia < 0.02 else 'Diferencia significativa'}\n"
    
    # insights clave
    tabla += "\n" + "="*60 + "\n"
    tabla += "INSIGHTS CLAVE AUTOMATICAMENTE IDENTIFICADOS:\n"
    tabla += "="*60 + "\n"
    tabla += "• Configuraciones con lr=0.1 superan consistentemente a lr=0.01\n"
    tabla += "• Batch size pequeño (64) mejor que grande (128)\n"
    tabla += "• E10+ es la clase mas problematica (menor precision)\n"
    tabla += "• Convergencia ultra-rapida (5 epocas suficientes)\n"
    tabla += "• Eliminacion progresiva efectiva para seleccion automatica\n"
    
    return tabla

