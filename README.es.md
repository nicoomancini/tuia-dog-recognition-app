
<div align="center">

# Reconocimiento de Razas de Perros

**Un pipeline integral de Visión por Computadora que combina búsqueda de imágenes similares, clasificación supervisada y detección de múltiples objetos para reconocer razas de perros.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C?logo=pytorch&logoColor=white)]()
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLOv8-111F68?logo=ultralytics&logoColor=white)]()
[![Gradio](https://img.shields.io/badge/Gradio-6.12.0-F97316?logo=gradio&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136.0-009688?logo=fastapi&logoColor=white)]()
[![pgvector](https://img.shields.io/badge/pgvector-PostgreSQL-4169E1?logo=postgresql&logoColor=white)]()
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)]()

</div>

**Idiomas:** [English](README.md) | Español

---

## Descripción general

Este proyecto implementa un sistema integral de Visión por Computadora para reconocer razas de perros a partir de imágenes.

En lugar de depender de un único modelo de clasificación, el proyecto explora e integra tres enfoques diferentes:

1. **Búsqueda por similitud visual**, utilizando embeddings de imágenes y recuperación vectorial.
2. **Clasificación supervisada de razas**, mediante transfer learning y una CNN personalizada entrenada desde cero.
3. **Detección y clasificación de múltiples objetos**, combinando YOLOv8 con los clasificadores de razas entrenados.

El proyecto incluye experimentación con modelos, evaluación cuantitativa, análisis de datos fuera de distribución (OOD) y una arquitectura de aplicación modular desarrollada con FastAPI, Gradio, PostgreSQL/pgvector y Docker.

---

## Resultados principales

| Área | Mejor resultado |
|---|---|
| Búsqueda de imágenes | **NDCG@10: 0.9646** con embeddings de ConvNeXt-Tiny |
| Clasificación basada en similitud | **97.14% de accuracy en validación** mediante weighted KNN con K=7 |
| Detección OOD | **ROC-AUC: 0.9243** |
| Clasificación supervisada | **97.00% de accuracy en test / 96.95% de F1** con ResNet18 fine-tuned |
| CNN personalizada | **83.57% de accuracy en test / 83.39% de F1**, entrenada desde cero |
| Experimentos de búsqueda de imágenes | **4 backbones × 2 métricas de distancia evaluadas** |

El proyecto también evalúa el pipeline completo de detección y clasificación, y analiza el impacto de las detecciones fallidas y del domain shift en el rendimiento de extremo a extremo.

---

## Arquitectura del sistema

La aplicación admite tres flujos de trabajo de Visión por Computadora independientes pero complementarios:

```text
                              IMAGEN DE ENTRADA
                                      │
                  ┌───────────────────┼───────────────────┐
                  │                   │                   │
                  ▼                   ▼                   ▼
          BÚSQUEDA POR         CLASIFICACIÓN       DETECCIÓN +
           SIMILITUD                                CLASIFICACIÓN
                  │                   │                   │
             ConvNeXt-Tiny        ResNet18 /           YOLOv8n
              Embeddings          Custom CNN              │
                  │                   │                   ▼
          Búsqueda vectorial      Softmax           Bounding Boxes
                  │                   │              de perros
          Weighted KNN        Raza + Confianza          │
                  │                                      │
                  ▼                                      ▼
          Raza o "Unknown"                       Recorte de cada
                                                    detección
                                                         │
                                                         ▼
                                               Clasificación de
                                                  cada perro
```

---

# Funcionalidades

## 1. Búsqueda por similitud visual

La primera etapa implementa un sistema de **Content-Based Image Retrieval (CBIR)**.

Cada imagen se transforma en un embedding de alta dimensionalidad utilizando una red neuronal preentrenada. Esta representación vectorial se utiliza para recuperar imágenes visualmente similares desde una base de datos vectorial.

La raza predicha se determina mediante una votación ponderada entre los vecinos más cercanos.

Además, se incorpora un umbral de similitud que permite rechazar imágenes fuera de la distribución conocida y devolver `"unknown"` en lugar de forzar una predicción poco confiable.

### Evaluación experimental

Se evaluaron cuatro backbones preentrenados:

- ResNet18
- ResNet50
- EfficientNet-B0
- ConvNeXt-Tiny

Se compararon dos métricas de distancia:

- Distancia euclidiana (L2).
- Similitud coseno.

La mejor configuración fue **ConvNeXt-Tiny con distancia euclidiana**, alcanzando:

- **NDCG@10: 0.9646**
- **97.14% de accuracy en validación** mediante votación ponderada con **K=7**.

El mecanismo de rechazo OOD alcanzó un **ROC-AUC de 0.9243**.

### Pipeline

```text
Imagen de entrada
       ↓
Extracción del embedding
       ↓
Búsqueda vectorial
       ↓
Top-K imágenes similares
       ↓
Votación ponderada de vecinos
       ↓
Predicción de raza o "Unknown"
```

### Captura de la aplicación

<div align="center">
  <img src="docs/screenshots/etapa1_similitud.png"
       alt="Etapa 1 - Búsqueda por similitud visual"
       width="1000"/>
</div>

---

## 2. Clasificación supervisada de razas

La segunda etapa se centra en la clasificación supervisada directa de **70 razas de perros**.

Se implementaron y evaluaron dos enfoques diferentes:

### ResNet18 fine-tuned

Se adaptó un modelo ResNet18 preentrenado con ImageNet al problema de clasificación de 70 clases.

Se reemplazó la capa final de clasificación y se realizó fine-tuning del modelo utilizando el dataset objetivo.

**Resultados en test:**

| Métrica | ResNet18 |
|---|---:|
| Accuracy | **97.00%** |
| Precision (Macro) | 97.16% |
| Recall (Macro) | 97.00% |
| Specificity | 99.96% |
| F1-Score (Macro) | **96.95%** |

### CNN personalizada entrenada desde cero

También se diseñó y entrenó una red neuronal convolucional personalizada, sin utilizar pesos preentrenados.

La arquitectura está compuesta por:

- Cuatro bloques convolucionales.
- Batch normalization.
- Activaciones ReLU.
- Max pooling.
- Adaptive average pooling.
- Una capa de clasificación fully connected con dropout.

El modelo se entrenó utilizando data augmentation, ponderación de clases, ajuste dinámico del learning rate y early stopping.

**Resultados en test:**

| Métrica | CNN personalizada |
|---|---:|
| Accuracy | **83.57%** |
| Precision (Macro) | 85.20% |
| Recall (Macro) | 83.57% |
| Specificity | 99.76% |
| F1-Score (Macro) | **83.39%** |

La comparación evidencia la ventaja en rendimiento del transfer learning y, al mismo tiempo, demuestra la capacidad de diseñar y entrenar una CNN multiclase desde cero.

### Pipeline

```text
Imagen de entrada
       ↓
Preprocesamiento
       ↓
Clasificador de razas
       ↓
Probabilidades Softmax
       ↓
Raza predicha + Confianza
```

### Captura de la aplicación

<div align="center">
  <img src="docs/screenshots/etapa2_clasificacion.png"
       alt="Etapa 2 - Clasificación supervisada"
       width="1000"/>
</div>

---

## 3. Detección de múltiples objetos y clasificación de razas

La tercera etapa amplía el sistema para trabajar con escenas más realistas que pueden contener uno o varios perros.

Se utiliza un detector **YOLOv8n** preentrenado para identificar perros en la imagen de entrada. Cada bounding box detectado se recorta y se procesa de manera independiente mediante el clasificador de razas.

Esto divide el problema en dos etapas:

1. **Localización de objetos:** detección de cada perro presente en la escena.
2. **Clasificación de razas:** asignación de una raza a cada instancia detectada.

### Pipeline

```text
Imagen de entrada
       ↓
Detección de perros con YOLOv8n
       ↓
Bounding Boxes
       ↓
Recorte de cada perro detectado
       ↓
Clasificación de razas
       ↓
Imagen anotada + Predicciones
```

El pipeline se evaluó en diferentes escenarios, incluyendo:

- Imágenes con un solo perro.
- Imágenes con múltiples perros en una misma escena.
- Imágenes sin perros.
- Imágenes con razas fuera de la distribución de entrenamiento.

### Evaluación de extremo a extremo

El pipeline integrado también permitió identificar limitaciones importantes.

Al evaluarlo sobre el conjunto de test, el rendimiento general se vio significativamente afectado por las detecciones fallidas del modelo YOLOv8n preentrenado con COCO.

El análisis identificó dos factores principales:

- **Cuello de botella en la detección:** el detector no logró identificar perros en una proporción considerable de las imágenes, particularmente en fotografías de primeros planos y con fondos de estudio.
- **Domain shift:** los clasificadores fueron entrenados con imágenes completas, pero durante la inferencia de extremo a extremo recibieron recortes generados por el detector.

Estos resultados permiten identificar líneas concretas de mejora, como realizar fine-tuning del detector con datos específicos de perros y entrenar los clasificadores utilizando imágenes recortadas de manera similar a las generadas por el detector.

### Captura de la aplicación

<div align="center">
  <img src="docs/screenshots/etapa3_deteccion.png"
       alt="Etapa 3 - Detección y clasificación"
       width="1000"/>
</div>

---

# Enfoque experimental

El proyecto se desarrolló como un estudio comparativo de Visión por Computadora, en lugar de depender de un único modelo.

Los experimentos incluyen:

- Comparación de **cuatro backbones preentrenados para extracción de embeddings**.
- Comparación de distancias **euclidiana y coseno** para la búsqueda de imágenes.
- Evaluación de la calidad de recuperación mediante **NDCG@10**.
- Optimización de la cantidad de vecinos para la clasificación mediante weighted KNN.
- Evaluación de un umbral de rechazo OOD mediante **análisis ROC**.
- Comparación entre **transfer learning y una CNN personalizada entrenada desde cero**.
- Evaluación mediante accuracy, macro precision, recall, specificity y macro F1-score.
- Evaluación del pipeline completo de detección y clasificación.
- Investigación de las causas de degradación del rendimiento en el sistema de extremo a extremo.

El objetivo no fue únicamente alcanzar una alta precisión en los modelos, sino también comprender cómo se comportan los distintos componentes al integrarse en una aplicación completa.

---

# Tecnologías

### Deep Learning y Visión por Computadora

- PyTorch
- torchvision
- Ultralytics YOLOv8
- OpenCV

### Machine Learning y Evaluación

- scikit-learn
- NumPy
- pandas

### Visualización

- matplotlib
- seaborn

### Backend y Despliegue

- FastAPI
- Uvicorn
- Docker
- Docker Compose

### Almacenamiento vectorial

- PostgreSQL
- pgvector
- Almacenamiento de embeddings basado en JSON

### Frontend

- Gradio

---

# Estructura del proyecto

El proyecto sigue una arquitectura modular que separa la API, los servicios de Visión por Computadora, el almacenamiento, la evaluación, la configuración y los componentes del frontend.

```text
tuia-dog-recognition-app/
├── src/
│   ├── app/
│   │   └── main.py
│   ├── frontend/
│   │   ├── app.py
│   │   └── gradio_ui.py
│   └── lib/
│       ├── api.py
│       ├── bootstrap.py
│       ├── config.py
│       ├── files.py
│       ├── schemas.py
│       ├── evaluation/
│       │   └── metrics.py
│       ├── visualization/
│       │   └── draw.py
│       ├── services/
│       │   ├── similarity_service.py
│       │   ├── classifier_service.py
│       │   ├── detection_service.py
│       │   └── task_manager.py
│       └── storage/
│           ├── embedding_store.py
│           └── pgvector_store.py
├── scripts/
│   ├── download_dataset.py
│   ├── build_index.py
│   └── train_classifier.py
├── data/
│   ├── dataset/
│   └── embeddings.json
├── models/
├── output/
├── docs/
│   └── screenshots/
├── informe.ipynb
├── etapa1_colab.ipynb
├── etapa2_colab.ipynb
├── etapa3_colab.ipynb
├── Trabajo Práctico N°2.pdf
├── requirements.txt
├── requirements-etapas.txt
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── .env.docker.example / .env.local.example
```

El repositorio incluye tanto la aplicación como el trabajo experimental que la respalda:

- **`src/`** contiene la aplicación modular, la API, los servicios de Visión por Computadora, el almacenamiento y el frontend.
- **`scripts/`** incluye herramientas para la preparación del dataset, la generación del índice vectorial y el entrenamiento de modelos.
- **`etapa1_colab.ipynb`, `etapa2_colab.ipynb` y `etapa3_colab.ipynb`** documentan los experimentos y evaluaciones de cada etapa del pipeline.
- **`models/`** contiene los checkpoints de los modelos entrenados.
- **`docs/`** contiene capturas de la aplicación y documentación complementaria.
- La configuración de Docker permite ejecutar el frontend, el backend y la base de datos vectorial como servicios independientes.

---

# Instalación y uso

## Requisitos

- Python 3.12
- Git
- Docker Desktop
- Cuenta de Kaggle

## Opción 1 — Ejecutar con Docker

```bash
docker compose build
docker compose up -d
```

Servicios disponibles:

- Frontend: `http://localhost:8080`
- Backend: `http://localhost:8000`
- PostgreSQL: `localhost:5432`

## Opción 2 — Entorno local

### 1. Clonar el repositorio

```bash
git clone <repository-url>
cd tuia-dog-recognition-app
```

### 2. Crear un entorno de Python

**Windows**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Como alternativa, utilizando `uv`:

```bash
uv venv --python 3.12 .venv
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

O utilizando `uv`:

```bash
uv pip install -r requirements.txt
```

### 4. Configurar las variables de entorno

**Linux / macOS**

```bash
cp .env.local.example src/.env
```

**Windows**

```bash
copy .env.local.example src\.env
```

### 5. Descargar el dataset

```bash
python scripts/download_dataset.py
```

Se requieren credenciales de la API de Kaggle.

Como alternativa, se puede descargar el dataset manualmente y colocar los directorios `train/`, `valid/` y `test/` dentro de `data/dataset`.

### 6. Iniciar la base de datos vectorial

Es necesario cuando `USE_PGVECTOR=true`:

```bash
docker compose up postgres -d
```

Cuando `USE_PGVECTOR=false`, la aplicación utiliza el almacenamiento local de embeddings en formato JSON.

### 7. Agregar los checkpoints de los modelos entrenados

Colocar los checkpoints dentro del directorio `models/`.

Configurar las rutas correspondientes mediante:

```text
RESNET18_MODEL_NAME
CNN_CUSTOM_MODEL_NAME
```

La aplicación admite modelos en formato `.pth` y `.onnx`.

### 8. Construir el índice vectorial

```bash
python scripts/build_index.py --split train
```

### 9. Iniciar el backend

```bash
cd src
uvicorn app.main:app --reload --port 8000
```

### 10. Iniciar el frontend

En otra terminal:

```bash
cd src
uvicorn frontend.app:app --port 8080
```

### 11. Verificar la aplicación

- Frontend: `http://localhost:8080`
- Health check del backend: `http://localhost:8000/health`

---

# Endpoints de la API

| Endpoint | Descripción |
|---|---|
| `POST /upload` | Subir una imagen |
| `POST /search` | Buscar imágenes visualmente similares |
| `POST /classify` | Clasificar una raza de perro |
| `POST /detect` | Detectar y clasificar uno o varios perros |
| `GET /status/{job_id}` | Consultar el estado de una tarea asíncrona |
| `GET /models` | Listar los modelos de embeddings disponibles |
| `GET /health` | Verificar el estado del backend |

Los endpoints asíncronos (`/search` y `/detect`) devuelven `HTTP 202` junto con un `job_id`.

```json
{
  "status": "done | inProgress | failed",
  "link": "url | none"
}
```

---

# Scripts útiles

```bash
# Descargar el dataset
python scripts/download_dataset.py

# Construir el índice vectorial
python scripts/build_index.py --split train

# Entrenar y evaluar un clasificador
python scripts/train_classifier.py --model resnet18_finetuned
```

---

# Configuración

La configuración de la aplicación se gestiona mediante variables de entorno.

Los principales parámetros incluyen:

- `EMBEDDING_MODEL`: modelo de embeddings utilizado para la búsqueda por similitud.
- `SIMILARITY_THRESHOLD`: umbral para el rechazo OOD.
- `TOP_K`: cantidad de vecinos más cercanos.
- `USE_PGVECTOR`: permite alternar entre PostgreSQL/pgvector y el almacenamiento local en JSON.
- `EMBEDDING_DIM`: dimensionalidad de los embeddings.
- `RESNET18_MODEL_NAME`: ruta al checkpoint de ResNet18.
- `CNN_CUSTOM_MODEL_NAME`: ruta al checkpoint de la CNN personalizada.

Al cambiar el modelo de embeddings, es necesario reconstruir el índice vectorial.

---

# Mejoras futuras

La evaluación de extremo a extremo permitió identificar distintas oportunidades de mejora:

- Realizar fine-tuning del detector con datos de bounding boxes específicos de perros.
- Entrenar los clasificadores utilizando recortes generados de manera similar a los del detector, para reducir el domain shift.
- Evaluar el sistema con datasets más diversos y representativos de escenarios reales.
- Optimizar la inferencia utilizando ONNX Runtime o TensorRT para un despliegue orientado a producción.

Estas mejoras se basan en las limitaciones identificadas durante la evaluación de extremo a extremo y no son requisitos para el funcionamiento del sistema actual.

---

# Alcance del proyecto

Este proyecto demuestra experiencia en:

- Desarrollo de pipelines de Visión por Computadora.
- Deep Learning con PyTorch.
- Transfer learning y fine-tuning.
- Diseño y entrenamiento de arquitecturas CNN personalizadas.
- Embeddings de imágenes y búsqueda por similitud vectorial.
- Detección de objetos con YOLO.
- Análisis de datos fuera de distribución (OOD).
- Comparación de modelos y evaluación cuantitativa.
- Desarrollo de APIs REST con FastAPI.
- Bases de datos vectoriales con pgvector.
- Aplicaciones contenerizadas con Docker.

---

## Documentación

El repositorio incluye notebooks experimentales y documentación técnica detallada sobre el entrenamiento de modelos, la evaluación, las métricas y las decisiones de implementación.

Parte de la documentación técnica ampliada está disponible actualmente en **español**, mientras que este README ofrece una descripción general en inglés de la arquitectura, los experimentos y los principales resultados.

<div align="center">

**Visión por Computadora · Deep Learning · Búsqueda de Imágenes · Detección de Objetos**

</div>
