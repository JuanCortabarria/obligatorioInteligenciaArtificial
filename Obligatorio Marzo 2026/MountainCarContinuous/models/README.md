# Visualización de modelos entrenados

Los archivos `.pkl` contienen los modelos entrenados utilizados en el proyecto. En particular, guardan las tablas Q aprendidas por los algoritmos de aprendizaje por refuerzo.

Se agregan imágenes `.png` generadas a partir de esos modelos. Estas visualizaciones permiten inspeccionar el comportamiento aprendido sin tener que ejecutar el código.

## Archivos incluidos

- `smoke_test.pkl`: modelo utilizado para prueba inicial.
- `q_learning_best.pkl`: mejor modelo entrenado con Q-Learning.
- `dyna_q_best.pkl`: mejor modelo entrenado con Dyna-Q.

## Smoke test

### Mejor valor Q por estado

Esta imagen muestra, para cada estado discretizado, el valor Q máximo entre todas las acciones posibles.

![Smoke test - Mejor valor Q](model_visualizations_png/smoke_test_max_q_heatmap.png)

### Política aprendida

Esta imagen muestra qué acción elige el modelo en cada estado, tomando la acción con mayor valor Q.

![Smoke test - Política aprendida](model_visualizations_png/smoke_test_policy.png)

## Q-Learning

### Mejor valor Q por estado

![Q-Learning - Mejor valor Q](model_visualizations_png/q_learning_best_max_q_heatmap.png)

### Política aprendida

![Q-Learning - Política aprendida](model_visualizations_png/q_learning_best_policy.png)

## Dyna-Q

### Mejor valor Q por estado

![Dyna-Q - Mejor valor Q](model_visualizations_png/dyna_q_best_max_q_heatmap.png)

### Política aprendida

![Dyna-Q - Política aprendida](model_visualizations_png/dyna_q_best_policy.png)

### Cobertura del modelo interno

Esta imagen muestra cuántas transiciones fueron aprendidas por el modelo interno de Dyna-Q para cada estado discretizado.

![Dyna-Q - Cobertura del modelo interno](model_visualizations_png/dyna_q_best_model_coverage.png)

## Nota

Los archivos `.pkl` se mantienen en el repositorio porque son los modelos entrenados. Las imágenes `.png` se agregan únicamente para facilitar la visualización desde GitHub.
