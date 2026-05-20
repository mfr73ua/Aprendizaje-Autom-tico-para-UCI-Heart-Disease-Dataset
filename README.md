# 🫀 Predicción de Riesgo Cardíaco con Privacidad, Explicabilidad y Equidad

> Trabajo final de la asignatura **Aprendizaje Avanzado** — Grado en Ingeniería en Inteligencia Artificial  
> Escuela Politécnica Superior, Universidad de Alicante — Curso 2025/2026

**Autores:** Marcos Francés Requena · Luna Camacho Boluda · Irene Fátima Robles Cano · Ángel Adrián Bustios Cancino

---

## 📋 Descripción

Sistema de predicción de riesgo cardíaco sobre el **UCI Heart Disease Dataset** (303 instancias, 13 variables clínicas) que aborda simultáneamente tres problemáticas del mundo real:

- **Privacidad** en entornos distribuidos → Aprendizaje Federado (FedAvg)
- **Escasez de datos y desbalance de clases** → SMOTETomek
- **Falta de transparencia** en los modelos → Explicabilidad SHAP

---

## 📁 Estructura del repositorio

```
.
├── codigo.ipynb                    # Notebook principal con todo el código
├── Informe.pdf                     # Memoria completa del proyecto
├── presentacion_diapositivas.pdf   # Diapositivas de la presentación
├── presentacion_diapositivas.pptx  # Diapositivas en formato editable
├── requirements.txt                # Dependencias del proyecto
├── images/                         # Gráficas y figuras generadas
└── tables/                         # Tablas de resultados exportadas
```

---

## 🗃️ Dataset

| Característica | Valor |
|---|---|
| Fuente | UCI ML Repository, `id=45` (Cleveland Clinic Foundation) |
| Instancias | 303 (297 tras eliminar nulos) |
| Variables | 13 clínicas + 1 objetivo |
| Tarea | Clasificación binaria: sano (0) / enfermedad (1) |
| Distribución | 54,1% clase 0 · 45,9% clase 1 |
| Licencia | CC BY 4.0 |

Las variables incluyen datos **numéricos continuos** (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`) y **categóricos** (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `thal`).

---

## 🔬 Metodología

### Preprocesamiento
- Eliminación de filas con valores nulos (`ca`: 4 nulos, `thal`: 2 nulos)
- División estratificada train/test (80/20, `random_state=42`)
- Imputación con **mediana** (numéricas) y **moda** (categóricas)
- Estandarización con `StandardScaler` y codificación con `OneHotEncoder(drop='first')`
- Balanceo con **SMOTETomek** exclusivamente sobre el conjunto de entrenamiento

### Modelos evaluados
| Modelo | Rol |
|---|---|
| Regresión Logística | Baseline de referencia |
| SVM (lineal, RBF, polinomial, sigmoide) | Modelos principales |
| Random Forest | Tratamiento de escasez + análisis SHAP |
| Gradient Boosting | Modelo alternativo |
| AdaBoost | Modelo alternativo |

### Aprendizaje Federado
Esquema con **K=10 clientes** (hospitales simulados), **R=15 rondas** de comunicación y agregación **FedAvg**. El balanceo con SMOTETomek se aplica localmente en cada cliente, sin compartir ninguna muestra.

### Explicabilidad
Análisis **SHAP** con `TreeExplainer` sobre el mejor Random Forest, generando importancia global (beeswarm + bar plots) y explicaciones locales por paciente (waterfall plots).

### Análisis de Fairness
Evaluación de equidad respecto al atributo sensible **sexo**, con métricas: Demographic Parity Difference, Equal Opportunity Difference, Equalized Odds Difference y Disparate Impact Ratio.

---

## 📊 Resultados principales

### Comparativa de modelos (GridSearch CV=5, SMOTETomek)

| Modelo | AUC-ROC | F1 | Recall | Precision | Accuracy |
|---|---|---|---|---|---|
| Baseline (LogReg) | 0,8906 | 0,7861 | 0,7511 | 0,8349 | 0,8140 |
| **SVM RBF** ⭐ | **0,9609** | **0,8727** | **0,8571** | **0,8889** | **0,8833** |
| SVM lineal | 0,9587 | 0,8727 | 0,8571 | 0,8889 | 0,8833 |
| SVM sigmoide | 0,9576 | 0,8727 | 0,8571 | 0,8889 | 0,8833 |
| SVM polinomial | 0,9386 | 0,8519 | 0,8214 | 0,8846 | 0,8667 |
| Random Forest | 0,9269 | 0,7692 | 0,7143 | 0,8333 | 0,8000 |
| Gradient Boosting | 0,9196 | 0,8000 | 0,7857 | 0,8148 | 0,8167 |
| AdaBoost | 0,9129 | 0,7778 | 0,7500 | 0,8077 | 0,8000 |

El **SVM con kernel RBF** es el mejor modelo (AUC-ROC = 0,9609), superando al baseline en más de **10 puntos porcentuales en Recall** — métrica prioritaria en diagnóstico médico.

### Variables más relevantes (SHAP)
1. `thal = 7.0` — Prueba de talio con defecto reversible
2. `cp = 4.0` — Dolor torácico asintomático
3. `ca` — Número de vasos coloreados por fluoroscopia
4. `oldpeak` — Depresión del segmento ST en ejercicio

Estas variables coinciden con los marcadores clínicos establecidos en la literatura médica.

### Fairness (atributo sensible: sexo)

| Indicador | Baseline | SVM RBF | Ideal |
|---|---|---|---|
| Demographic Parity Difference | 0,4801 | 0,4275 | ↓ 0 |
| Equal Opportunity Difference | 0,1733 | 0,1600 | ↓ 0 |
| Disparate Impact Ratio | 0,1798 | 0,2697 | ↑ 1 |

El análisis revela un **sesgo sistemático por sexo** (DIRatio muy por debajo del umbral de 0,80), consecuencia directa de la sobrerrepresentación masculina en los datos (68%).

---

## ⚙️ Instalación

```bash
git clone <url-del-repositorio>
cd <nombre-del-repositorio>
pip install -r requirements.txt
```

### Dependencias principales
- Python 3.12
- scikit-learn 1.6
- imbalanced-learn 0.13
- shap 0.47
- numpy 2.2
- pandas, matplotlib, seaborn
- ucimlrepo (descarga automática del dataset)

---

## 🚀 Uso

Ejecutar el notebook `codigo.ipynb` en orden. El dataset se descarga automáticamente mediante `ucimlrepo`:

```python
from ucimlrepo import fetch_ucirepo
heart_disease = fetch_ucirepo(id=45)
```

Los resultados (gráficas y tablas) se guardan en las carpetas `images/` y `tables/`.

---

## 📌 Conclusiones clave

- La **complejidad del modelo no es el factor determinante** con datasets pequeños: la diferencia entre el mejor y peor modelo es inferior a 5 puntos en AUC-ROC.
- **SMOTETomek integrado en el esquema federado** es compatible con la preservación de privacidad, al realizarse el balanceo localmente.
- La **privacidad diferencial moderada** (ε ≥ 1) es viable sin degradación significativa del rendimiento.
- Las **métricas globales son insuficientes** para validar sistemas de apoyo al diagnóstico: la evaluación desagregada por grupos sensibles debe ser parte del protocolo estándar.

---
