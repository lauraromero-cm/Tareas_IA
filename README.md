# Tarea IA 2 - Clasificación de Ratings de Videojuegos

Sistema de entrenamiento competitivo paralelo para clasificación de ratings de videojuegos usando Regresión Logística y SVM.

## Requisitos

- Python 3.12.7 (o compatible)
- CUDA (opcional, para GPU)

## Instalación

### 1. Crear y activar entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn pyyaml kagglehub
```

O con el archivo de requirements:

```bash
pip install -r requirements.txt
```

## Configuración

El proyecto usa `config.yaml` para definir hiperparámetros de entrenamiento. Este archivo ya está incluido y contiene:

- 3 configuraciones competitivas de Regresión Logística
- 3 configuraciones competitivas de SVM
- Explicación detallada de cada hiperparámetro

## Ejecución

### Ejecutar entrenamiento completo

```bash
python main.py
```
