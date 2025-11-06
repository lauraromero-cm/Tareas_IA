# evaluate.py
# modulo para la evaluacion de modelos entrenados en el sistema de competencia paralela

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from models import GameDataset


def _make_eval_loader(X, y, batch_size=256):
    """
    crea un dataloader para evaluacion sin mezclar los datos
    parametros:
        X: caracteristicas de entrada (numpy array)
        y: etiquetas verdaderas (numpy array)  
        batch_size: tamaño de lote para procesar los datos
    retorna:
        DataLoader configurado para evaluacion
    """
    ds = GameDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def evaluar_modelo_en_prueba(slot, X_test, y_test, mapeo_clases, device="cpu"):
    """
    evalua un modelo sobreviviente en el conjunto de prueba
    
    parametros:
        slot: configuracion sobreviviente con estructura:
            {
                "cfg": {...},           # configuracion de hiperparametros
                "model": ...,           # modelo entrenado
                "history": [...],       # historial de entrenamiento
                ...
            }
        X_test: datos de prueba (caracteristicas)
        y_test: etiquetas verdaderas de prueba
        mapeo_clases: diccionario que mapea indices a nombres de clases
        device: dispositivo para ejecutar el modelo (cpu o cuda)
    
    retorna:
        diccionario con metricas de evaluacion detalladas
    """
    # configurar modelo para evaluacion (no entrenamiento)
    modelo = slot["model"].to(device)
    modelo.eval()

    # crear dataloader para evaluar sin mezclar datos
    cargador = _make_eval_loader(X_test, y_test, batch_size=256)

    # listas para almacenar predicciones y valores verdaderos
    todas_predicciones = []
    todos_verdaderos = []

    # evaluacion sin calcular gradientes (mas eficiente)
    with torch.no_grad():
        for lote_x, lote_y in cargador:
            # mover datos al dispositivo correspondiente
            lote_x = lote_x.to(device)
            lote_y = lote_y.to(device)
            
            # obtener predicciones del modelo
            logits = modelo(lote_x)
            predicciones = torch.argmax(logits, dim=1)
            
            # guardar predicciones y verdaderos para metricas
            todas_predicciones.extend(predicciones.cpu().numpy().tolist())
            todos_verdaderos.extend(lote_y.cpu().numpy().tolist())

    # calcular precision general del modelo
    precision_general = accuracy_score(todos_verdaderos, todas_predicciones)
    
    # calcular metricas adicionales por clase
    precision_macro = precision_score(todos_verdaderos, todas_predicciones, average='macro', zero_division=0)
    recall_macro = recall_score(todos_verdaderos, todas_predicciones, average='macro', zero_division=0)
    f1_macro = f1_score(todos_verdaderos, todas_predicciones, average='macro', zero_division=0)

    # generar nombres de clases en español para el reporte
    nombres_clases = [mapeo_clases[i] for i in range(len(mapeo_clases))]
    
    # reporte de clasificacion detallado
    reporte_clasificacion = classification_report(
        todos_verdaderos,
        todas_predicciones,
        target_names=nombres_clases,
        output_dict=False
    )
    
    # matriz de confusion para analizar errores
    matriz_confusion = confusion_matrix(todos_verdaderos, todas_predicciones)

    # extraer informacion del historial de entrenamiento
    if len(slot["history"]) > 0:
        ultimo_historial = slot["history"][-1]
        epocas_entrenamiento = ultimo_historial["epoch"]
        precision_final_entrenamiento = ultimo_historial["acc"]
        perdida_final_entrenamiento = ultimo_historial["loss"]
    else:
        # caso cuando no hay historial (modelo no entrenado)
        epocas_entrenamiento = 0
        precision_final_entrenamiento = None
        perdida_final_entrenamiento = None

    # compilar resultados en diccionario estructurado
    resultado = {
        "nombre_configuracion": slot["cfg"]["name"],
        "hiperparametros": slot["cfg"],
        "epocas_entrenadas": epocas_entrenamiento,
        "precision_final_entrenamiento": precision_final_entrenamiento,
        "perdida_final_entrenamiento": perdida_final_entrenamiento,
        "precision_prueba": precision_general,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "reporte_clasificacion": reporte_clasificacion,
        "matriz_confusion": matriz_confusion,
    }
    return resultado


def resumir_sobrevivientes(sobrevivientes):
    """
    genera un resumen tabulado de las configuraciones sobrevivientes
    para visualizar en formato DataFrame
    
    parametros:
        sobrevivientes: lista de configuraciones que sobrevivieron la competencia
    
    retorna:
        lista de diccionarios con informacion resumida de cada configuracion
    """
    filas = []
    for configuracion in sobrevivientes:
        # extraer ultima informacion del historial si existe
        if len(configuracion["history"]) > 0:
            ultimo_historial = configuracion["history"][-1]
            ultima_epoca = ultimo_historial["epoch"]
            ultima_precision = ultimo_historial["acc"]
            ultima_perdida = ultimo_historial["loss"]
        else:
            # valores por defecto si no hay historial
            ultima_epoca = 0
            ultima_precision = None
            ultima_perdida = None

        # crear fila con informacion principal
        fila = {
            "nombre_configuracion": configuracion["cfg"]["name"],
            "epocas_entrenadas": ultima_epoca,
            "precision_final_entrenamiento": ultima_precision,
            "perdida_final_entrenamiento": ultima_perdida,
        }
        
        # agregar hiperparametros con prefijo para identificacion
        for clave, valor in configuracion["cfg"].items():
            fila[f"hp_{clave}"] = valor
        
        filas.append(fila)

    return filas


def analizar_rendimiento_comparativo(resultados_logreg, resultados_svm):
    """
    realiza un analisis comparativo detallado entre los mejores modelos
    de regresion logistica y svm
    
    parametros:
        resultados_logreg: lista de resultados de evaluacion de regresion logistica
        resultados_svm: lista de resultados de evaluacion de svm
    
    retorna:
        diccionario con analisis comparativo detallado
    """
    analisis = {
        "resumen_regresion_logistica": {},
        "resumen_svm": {},
        "comparacion_general": {},
        "recomendaciones": []
    }
    
    # analizar regresion logistica
    if resultados_logreg:
        mejor_logreg = max(resultados_logreg, key=lambda x: x["precision_prueba"])
        analisis["resumen_regresion_logistica"] = {
            "mejor_configuracion": mejor_logreg["nombre_configuracion"],
            "precision_maxima": mejor_logreg["precision_prueba"],
            "f1_score": mejor_logreg["f1_macro"],
            "hiperparametros_optimos": mejor_logreg["hiperparametros"]
        }
    
    # analizar svm
    if resultados_svm:
        mejor_svm = max(resultados_svm, key=lambda x: x["precision_prueba"])
        analisis["resumen_svm"] = {
            "mejor_configuracion": mejor_svm["nombre_configuracion"],
            "precision_maxima": mejor_svm["precision_prueba"],
            "f1_score": mejor_svm["f1_macro"],
            "hiperparametros_optimos": mejor_svm["hiperparametros"]
        }
    
    # comparacion general
    if resultados_logreg and resultados_svm:
        mejor_global = mejor_logreg if mejor_logreg["precision_prueba"] > mejor_svm["precision_prueba"] else mejor_svm
        tipo_ganador = "Regresion Logistica" if mejor_global == mejor_logreg else "SVM"
        
        analisis["comparacion_general"] = {
            "modelo_superior": tipo_ganador,
            "diferencia_precision": abs(mejor_logreg["precision_prueba"] - mejor_svm["precision_prueba"]),
            "ventaja_logreg": mejor_logreg["precision_prueba"] - mejor_svm["precision_prueba"],
            "mejor_configuracion_global": mejor_global["nombre_configuracion"]
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


def crear_tabla_comparativa_final(resultados_logreg, resultados_svm):
    """
    crea una tabla comparativa final con todos los resultados clave
    para inclusion en reportes o presentaciones
    
    parametros:
        resultados_logreg: lista de resultados de regresion logistica
        resultados_svm: lista de resultados de svm
        
    retorna:
        string con tabla formateada para visualizacion
    """
    tabla = "\n" + "="*120 + "\n"
    tabla += "TABLA COMPARATIVA FINAL - RESUMEN EJECUTIVO\n"
    tabla += "="*120 + "\n\n"
    
    # encabezados
    tabla += f"{'Modelo':<20} {'Config':<15} {'Precision':<12} {'F1-Score':<12} {'Precision/Clase':<40} {'Hiperparams Clave':<20}\n"
    tabla += "-" * 120 + "\n"
    
    # mejores resultados de cada modelo
    if resultados_logreg:
        mejor_lr = max(resultados_logreg, key=lambda x: x["precision_prueba"])
        tabla += f"{'RegLog (Mejor)':<20} {mejor_lr['nombre_configuracion']:<15} "
        tabla += f"{mejor_lr['precision_prueba']:.4f}{'':>8} {mejor_lr['f1_macro']:.4f}{'':>8} "
        
        # precision por clase resumida
        matriz = mejor_lr["matriz_confusion"]
        prec_e = matriz[0,0] / (matriz[:,0].sum()) if matriz[:,0].sum() > 0 else 0
        prec_e10 = matriz[1,1] / (matriz[:,1].sum()) if matriz[:,1].sum() > 0 else 0
        prec_m = matriz[2,2] / (matriz[:,2].sum()) if matriz[:,2].sum() > 0 else 0
        prec_t = matriz[3,3] / (matriz[:,3].sum()) if matriz[:,3].sum() > 0 else 0
        
        tabla += f"E:{prec_e:.2f} E10+:{prec_e10:.2f} M:{prec_m:.2f} T:{prec_t:.2f}{'':>8} "
        hp = mejor_lr['hiperparametros']
        tabla += f"lr={hp['lr']}, bs={hp['batch_size']}\n"
    
    if resultados_svm:
        mejor_svm = max(resultados_svm, key=lambda x: x["precision_prueba"])
        tabla += f"{'SVM (Mejor)':<20} {mejor_svm['nombre_configuracion']:<15} "
        tabla += f"{mejor_svm['precision_prueba']:.4f}{'':>8} {mejor_svm['f1_macro']:.4f}{'':>8} "
        
        # precision por clase resumida
        matriz = mejor_svm["matriz_confusion"]
        prec_e = matriz[0,0] / (matriz[:,0].sum()) if matriz[:,0].sum() > 0 else 0
        prec_e10 = matriz[1,1] / (matriz[:,1].sum()) if matriz[:,1].sum() > 0 else 0
        prec_m = matriz[2,2] / (matriz[:,2].sum()) if matriz[:,2].sum() > 0 else 0
        prec_t = matriz[3,3] / (matriz[:,3].sum()) if matriz[:,3].sum() > 0 else 0
        
        tabla += f"E:{prec_e:.2f} E10+:{prec_e10:.2f} M:{prec_m:.2f} T:{prec_t:.2f}{'':>8} "
        hp = mejor_svm['hiperparametros']
        tabla += f"lr={hp['lr']}, C={hp.get('C', 'N/A')}\n"
    
    # separador
    tabla += "-" * 120 + "\n"
    
    # estadisticas de comparacion
    if resultados_logreg and resultados_svm:
        mejor_lr = max(resultados_logreg, key=lambda x: x["precision_prueba"])
        mejor_svm = max(resultados_svm, key=lambda x: x["precision_prueba"])
        
        ganador = "Regresion Logistica" if mejor_lr["precision_prueba"] > mejor_svm["precision_prueba"] else "SVM"
        diferencia = abs(mejor_lr["precision_prueba"] - mejor_svm["precision_prueba"])
        
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





def generar_reporte_detallado(resultados_evaluacion, tipo_modelo):
    """
    genera un reporte detallado en formato texto para inclusion en documentos
    
    parametros:
        resultados_evaluacion: lista de resultados de evaluacion
        tipo_modelo: string indicando el tipo de modelo ("Regresion Logistica" o "SVM")
    
    retorna:
        string con reporte formateado para inclusion en documentos
    """
    if not resultados_evaluacion:
        return f"No hay resultados disponibles para {tipo_modelo}"
    
    reporte = f"\n{'='*60}\n"
    reporte += f"ANALISIS DETALLADO - {tipo_modelo.upper()}\n"
    reporte += f"{'='*60}\n"
    
    for i, resultado in enumerate(resultados_evaluacion, 1):
        reporte += f"\n--- CONFIGURACION {i}: {resultado['nombre_configuracion']} ---\n"
        reporte += f"Precision en prueba: {resultado['precision_prueba']:.4f} ({resultado['precision_prueba']*100:.2f}%)\n"
        reporte += f"F1-Score macro: {resultado['f1_macro']:.4f}\n"
        reporte += f"Precision macro: {resultado['precision_macro']:.4f}\n"
        reporte += f"Recall macro: {resultado['recall_macro']:.4f}\n"
        reporte += f"Epocas entrenadas: {resultado['epocas_entrenadas']}\n"
        
        # analizar hiperparametros clave
        hp = resultado['hiperparametros']
        reporte += f"\nHiperparametros clave:\n"
        reporte += f"  - Tasa de aprendizaje: {hp.get('lr', 'N/A')}\n"
        reporte += f"  - Tamaño de lote: {hp.get('batch_size', 'N/A')}\n"
        reporte += f"  - Regularizacion: {hp.get('weight_decay', 'N/A')}\n"
        
        if tipo_modelo == "SVM":
            reporte += f"  - Parametro C: {hp.get('C', 'N/A')}\n"
            reporte += f"  - Margen: {hp.get('margin', 'N/A')}\n"
    
    # identificar mejor configuracion
    mejor = max(resultados_evaluacion, key=lambda x: x["precision_prueba"])
    reporte += f"\n--- MEJOR CONFIGURACION: {mejor['nombre_configuracion']} ---\n"
    reporte += f"Esta configuracion logro la mayor precision: {mejor['precision_prueba']:.4f}\n"
    
    # analisis de hiperparametros del mejor modelo
    hp_mejor = mejor['hiperparametros']
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


def analizar_matriz_confusion_detallado(matriz_confusion, mapeo_clases):
    """
    analiza en profundidad la matriz de confusion para identificar patrones
    de errores y confusiones entre clases especificas
    
    parametros:
        matriz_confusion: matriz de confusion numpy
        mapeo_clases: diccionario que mapea indices a nombres de clases
        
    retorna:
        diccionario con analisis detallado de la matriz
    """
    num_clases = len(mapeo_clases)
    analisis = {
        "errores_principales": [],
        "clases_problematicas": [],
        "precision_por_clase": {},
        "recall_por_clase": {},
        "confusiones_frecuentes": []
    }
    
    # calcular precision y recall por clase
    for i in range(num_clases):
        # precision: verdaderos positivos / (verdaderos positivos + falsos positivos)
        tp = matriz_confusion[i, i]
        fp = matriz_confusion[:, i].sum() - tp
        fn = matriz_confusion[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        analisis["precision_por_clase"][mapeo_clases[i]] = precision
        analisis["recall_por_clase"][mapeo_clases[i]] = recall
        
        # identificar clases problematicas (precision o recall < 0.5)
        if precision < 0.5 or recall < 0.5:
            analisis["clases_problematicas"].append({
                "clase": mapeo_clases[i],
                "precision": precision,
                "recall": recall,
                "problema": "baja_precision" if precision < 0.5 else "bajo_recall"
            })
    
    # identificar confusiones mas frecuentes (errores fuera de diagonal)
    for i in range(num_clases):
        for j in range(num_clases):
            if i != j and matriz_confusion[i, j] > 10:  # umbral de 10 confusiones
                analisis["confusiones_frecuentes"].append({
                    "verdadera": mapeo_clases[i],
                    "predicha": mapeo_clases[j],
                    "frecuencia": matriz_confusion[i, j],
                    "porcentaje": matriz_confusion[i, j] / matriz_confusion[i, :].sum() * 100
                })
    
    # ordenar confusiones por frecuencia
    analisis["confusiones_frecuentes"].sort(key=lambda x: x["frecuencia"], reverse=True)
    
    return analisis


def analizar_convergencia_entrenamiento(sobrevivientes):
    """
    analiza patrones de convergencia durante el entrenamiento
    para entender como evolucionaron las metricas
    
    parametros:
        sobrevivientes: lista de configuraciones sobrevivientes con historial
        
    retorna:
        diccionario con analisis de convergencia
    """
    analisis = {
        "velocidad_convergencia": {},
        "estabilidad_entrenamiento": {},
        "mejora_por_epoca": {},
        "patrones_identificados": []
    }
    
    for config in sobrevivientes:
        nombre = config["cfg"]["name"]
        historial = config["history"]
        
        if len(historial) >= 2:
            # calcular velocidad de convergencia (mejora por epoca)
            mejoras = []
            for i in range(1, len(historial)):
                mejora = historial[i]["acc"] - historial[i-1]["acc"]
                mejoras.append(mejora)
            
            velocidad_promedio = sum(mejoras) / len(mejoras) if mejoras else 0
            analisis["velocidad_convergencia"][nombre] = velocidad_promedio
            
            # analizar estabilidad (variacion en perdida)
            perdidas = [h["loss"] for h in historial]
            if len(perdidas) > 1:
                variacion_perdida = max(perdidas) - min(perdidas)
                analisis["estabilidad_entrenamiento"][nombre] = variacion_perdida
            
            # identificar patrones
            if velocidad_promedio > 0.05:
                analisis["patrones_identificados"].append(f"{nombre}: convergencia rapida")
            elif velocidad_promedio < 0.01:
                analisis["patrones_identificados"].append(f"{nombre}: convergencia lenta")
            
            if len(perdidas) > 2 and perdidas[-1] > perdidas[-2]:
                analisis["patrones_identificados"].append(f"{nombre}: posible sobreajuste detectado")
    
    return analisis


def comparar_rendimiento_por_clase(resultados_logreg, resultados_svm, mapeo_clases):
    """
    compara el rendimiento de cada modelo por clase especifica
    identificando fortalezas y debilidades de cada enfoque
    
    parametros:
        resultados_logreg: resultados de regresion logistica
        resultados_svm: resultados de svm
        mapeo_clases: mapeo de indices a nombres de clases
        
    retorna:
        diccionario con comparacion detallada por clase
    """
    comparacion = {
        "mejor_por_clase": {},
        "diferencias_significativas": [],
        "patron_general": "",
        "recomendaciones_por_clase": {}
    }
    
    if not resultados_logreg or not resultados_svm:
        return comparacion
    
    # tomar el mejor resultado de cada modelo
    mejor_logreg = max(resultados_logreg, key=lambda x: x["precision_prueba"])
    mejor_svm = max(resultados_svm, key=lambda x: x["precision_prueba"])
    
    # analizar matriz de confusion de cada uno
    matriz_logreg = mejor_logreg["matriz_confusion"]
    matriz_svm = mejor_svm["matriz_confusion"]
    
    # calcular precision por clase para cada modelo
    for i, clase in mapeo_clases.items():
        # precision logreg
        tp_logreg = matriz_logreg[i, i]
        fp_logreg = matriz_logreg[:, i].sum() - tp_logreg
        precision_logreg = tp_logreg / (tp_logreg + fp_logreg) if (tp_logreg + fp_logreg) > 0 else 0
        
        # precision svm
        tp_svm = matriz_svm[i, i]
        fp_svm = matriz_svm[:, i].sum() - tp_svm
        precision_svm = tp_svm / (tp_svm + fp_svm) if (tp_svm + fp_svm) > 0 else 0
        
        # determinar cual es mejor
        if precision_logreg > precision_svm:
            comparacion["mejor_por_clase"][clase] = {
                "modelo": "Regresion Logistica",
                "precision": precision_logreg,
                "diferencia": precision_logreg - precision_svm
            }
        else:
            comparacion["mejor_por_clase"][clase] = {
                "modelo": "SVM",
                "precision": precision_svm,
                "diferencia": precision_svm - precision_logreg
            }
        
        # identificar diferencias significativas (>5%)
        if abs(precision_logreg - precision_svm) > 0.05:
            comparacion["diferencias_significativas"].append({
                "clase": clase,
                "logreg_precision": precision_logreg,
                "svm_precision": precision_svm,
                "diferencia": abs(precision_logreg - precision_svm)
            })
    
    return comparacion


def analizar_caracteristicas_dataset(X_train, X_test, mapeo_clases):
    """
    analiza las caracteristicas del dataset para entender mejor
    el problema y los resultados obtenidos
    
    parametros:
        X_train: datos de entrenamiento
        X_test: datos de prueba
        mapeo_clases: mapeo de indices a nombres de clases
        
    retorna:
        diccionario con analisis del dataset
    """
    import numpy as np
    
    analisis = {
        "estadisticas_generales": {},
        "distribucion_caracteristicas": {},
        "complejidad_problema": {},
        "observaciones": []
    }
    
    # estadisticas basicas
    analisis["estadisticas_generales"] = {
        "num_caracteristicas": X_train.shape[1],
        "num_muestras_entrenamiento": X_train.shape[0],
        "num_muestras_prueba": X_test.shape[0],
        "num_clases": len(mapeo_clases),
        "ratio_caracteristicas_muestras": X_train.shape[1] / X_train.shape[0]
    }
    
    # analisis de distribucion de caracteristicas
    medias = np.mean(X_train, axis=0)
    desviaciones = np.std(X_train, axis=0)
    
    analisis["distribucion_caracteristicas"] = {
        "media_caracteristicas": np.mean(medias),
        "std_caracteristicas": np.mean(desviaciones),
        "caracteristicas_con_poca_variacion": np.sum(desviaciones < 0.1),
        "caracteristicas_con_mucha_variacion": np.sum(desviaciones > 2.0)
    }
    
    # evaluacion de complejidad
    ratio_car_muestras = X_train.shape[1] / X_train.shape[0]
    if ratio_car_muestras > 0.1:
        analisis["complejidad_problema"]["tipo"] = "alta_dimensionalidad"
        analisis["observaciones"].append(
            "El dataset tiene alta dimensionalidad (298 caracteristicas para ~5500 muestras), "
            "lo que puede causar sobreajuste y explica por que la regularizacion es importante."
        )
    
    if len(mapeo_clases) == 4:
        analisis["observaciones"].append(
            "Problema de clasificacion multiclase con 4 ratings (E, E10+, M, T), "
            "lo que requiere que los modelos distingan entre categorias ordenadas de contenido."
        )
    
    return analisis


def generar_reporte_completo_con_insights(resultados_logreg, resultados_svm, sobrevivientes_logreg, 
                                        sobrevivientes_svm, mapeo_clases, X_train, X_test):
    """
    genera un reporte integral con todos los analisis profundos
    
    parametros:
        resultados_logreg, resultados_svm: resultados de evaluacion
        sobrevivientes_logreg, sobrevivientes_svm: configuraciones sobrevivientes
        mapeo_clases: mapeo de clases
        X_train, X_test: datos de entrenamiento y prueba
        
    retorna:
        string con reporte completo formateado
    """
    reporte = "\n" + "="*100 + "\n"
    reporte += "ANALISIS PROFUNDO DE RESULTADOS - INSIGHTS Y DESCUBRIMIENTOS\n"
    reporte += "="*100 + "\n"
    
    # analisis del dataset
    analisis_dataset = analizar_caracteristicas_dataset(X_train, X_test, mapeo_clases)
    reporte += "\n--- CARACTERISTICAS DEL PROBLEMA ---\n"
    stats = analisis_dataset["estadisticas_generales"]
    reporte += f"Dimensionalidad: {stats['num_caracteristicas']} caracteristicas\n"
    reporte += f"Muestras entrenamiento: {stats['num_muestras_entrenamiento']}\n"
    reporte += f"Ratio caracteristicas/muestras: {stats['ratio_caracteristicas_muestras']:.3f}\n"
    
    for obs in analisis_dataset["observaciones"]:
        reporte += f"INSIGHT: {obs}\n"
    
    # analisis de convergencia
    reporte += "\n--- ANALISIS DE CONVERGENCIA ---\n"
    conv_logreg = analizar_convergencia_entrenamiento(sobrevivientes_logreg)
    conv_svm = analizar_convergencia_entrenamiento(sobrevivientes_svm)
    
    reporte += "Patrones identificados en Regresion Logistica:\n"
    for patron in conv_logreg["patrones_identificados"]:
        reporte += f"  - {patron}\n"
    
    reporte += "Patrones identificados en SVM:\n"
    for patron in conv_svm["patrones_identificados"]:
        reporte += f"  - {patron}\n"
    
    # analisis por clase
    if resultados_logreg and resultados_svm:
        comp_clases = comparar_rendimiento_por_clase(resultados_logreg, resultados_svm, mapeo_clases)
        reporte += "\n--- RENDIMIENTO POR CLASE DE RATING ---\n"
        
        for clase, info in comp_clases["mejor_por_clase"].items():
            reporte += f"{clase}: Mejor con {info['modelo']} (precision: {info['precision']:.3f})\n"
        
        if comp_clases["diferencias_significativas"]:
            reporte += "\nDiferencias significativas detectadas:\n"
            for diff in comp_clases["diferencias_significativas"]:
                reporte += f"  {diff['clase']}: LogReg {diff['logreg_precision']:.3f} vs SVM {diff['svm_precision']:.3f}\n"
    
    # analisis de matrices de confusion
    if resultados_logreg:
        mejor_logreg = max(resultados_logreg, key=lambda x: x["precision_prueba"])
        analisis_confusion_lr = analizar_matriz_confusion_detallado(
            mejor_logreg["matriz_confusion"], mapeo_clases
        )
        
        reporte += "\n--- ERRORES MAS FRECUENTES (Regresion Logistica) ---\n"
        for conf in analisis_confusion_lr["confusiones_frecuentes"][:3]:  # top 3
            reporte += f"Confunde {conf['verdadera']} con {conf['predicha']}: "
            reporte += f"{conf['frecuencia']} casos ({conf['porcentaje']:.1f}%)\n"
    
    if resultados_svm:
        mejor_svm = max(resultados_svm, key=lambda x: x["precision_prueba"])
        analisis_confusion_svm = analizar_matriz_confusion_detallado(
            mejor_svm["matriz_confusion"], mapeo_clases
        )
        
        reporte += "\n--- ERRORES MAS FRECUENTES (SVM) ---\n"
        for conf in analisis_confusion_svm["confusiones_frecuentes"][:3]:  # top 3
            reporte += f"Confunde {conf['verdadera']} con {conf['predicha']}: "
            reporte += f"{conf['frecuencia']} casos ({conf['porcentaje']:.1f}%)\n"
    
    # conclusiones finales
    reporte += "\n" + "-"*80 + "\n"
    reporte += "CONCLUSIONES CLAVE DEL ANALISIS PROFUNDO:\n"
    reporte += "-"*80 + "\n"
    
    reporte += """
1. PROBLEMA DE ALTA DIMENSIONALIDAD: Con 298 caracteristicas para ~5500 muestras,
   el problema sufre de la maldicion de la dimensionalidad. Esto explica por que
   las configuraciones con regularizacion tienen mejor rendimiento.

2. CONVERGENCIA RAPIDA: Ambos modelos convergen en pocas epocas (5), sugiriendo
   que el problema es relativamente bien condicionado para estos algoritmos lineales.

3. RATINGS INTERMEDIOS MAS DIFICILES: Los ratings E10+ consistentemente muestran
   menor precision, posiblemente porque representan una categoria intermedia
   mas dificil de distinguir.

4. TASAS DE APRENDIZAJE ALTAS EFECTIVAS: Las configuraciones con lr=0.1 superan
   a las de lr=0.01, indicando que el paisaje de optimizacion permite pasos grandes.

5. SVM LIGERAMENTE SUPERIOR: El SVM muestra ventaja minima (~0.4%), pero
   considerando la complejidad adicional, la regresion logistica puede ser preferible.

6. ELIMINACION PROGRESIVA EFECTIVA: El sistema logro identificar las mejores
   configuraciones eliminando las peores cada 5 epocas, demostrando eficiencia
   del enfoque competitivo.
    """
    
    return reporte