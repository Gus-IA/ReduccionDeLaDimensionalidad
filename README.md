# 🧠 Reducción de Dimensionalidad con Python y Scikit-learn

Este proyecto explora diferentes técnicas de reducción de dimensionalidad utilizando `NumPy`, `Matplotlib` y `Scikit-learn`. Se aplican conceptos de análisis de componentes principales (PCA) y otros métodos como `LLE`, `Isomap`, `MDS` y `t-SNE`, sobre datos sintéticos y el dataset MNIST.

---

## 📚 Contenido

### 1. **Visualización y PCA desde cero**
- Se generan datos 3D simulando una curva con ruido.
- Se visualizan en 3D y luego se reduce a 2D mediante PCA hecho "a mano" con `SVD` de `numpy`.
- Se reconstruyen los datos y se analiza la pérdida de información.

### 2. **PCA con `scikit-learn`**
- Se utiliza `PCA` de `sklearn` para realizar la reducción de dimensionalidad.
- Se demuestra cómo estimar el número mínimo de dimensiones necesarias para conservar un 95% de la varianza.

### 3. **Dataset MNIST**
- Se descarga el dataset de dígitos escritos a mano (784 dimensiones).
- Se aplica `PCA` para reducir las dimensiones mientras se conserva el 95% de la varianza.
- Se visualiza la comparación entre imágenes originales y comprimidas.

### 4. **Técnicas avanzadas de reducción no lineal**
- `LLE (Locally Linear Embedding)`
- `MDS (Multidimensional Scaling)`
- `Isomap`
- `t-SNE`

Se aplican al conjunto de datos suizo (`Swiss Roll`) generado con `make_swiss_roll`, demostrando cómo se "desenrolla" un dataset no lineal.

---

## 🧪 Requisitos

Instala las dependencias con:

```bash
pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
