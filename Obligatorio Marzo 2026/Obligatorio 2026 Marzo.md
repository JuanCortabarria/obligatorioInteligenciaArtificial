## Contexto del problema

La empresa **Red Destination™** los ha contratado para formar parte del desarrollo del próximo rover **“Out for Delivery”**, que será enviado a Marte.

En su rol, estarán encargados de implementar el agente inteligente responsable de su control autónomo.

---

## Objetivos

En **Red Destination™**, se espera que demuestren su expertise en la aplicación de técnicas de **Q-Learning** y **Minimax/Expectimax** en los dos proyectos presentados:

1. **Proyecto LOST**: *Learning-based Orientation and Steering for Traversal*.
2. **Proyecto MATE**: *Martian Adversarial Tactics Engine*.

---

## Proyecto LOST

### Learning-based Orientation and Steering for Traversal

Los ingenieros han logrado que el rover **“Out for Delivery”** aterrice exitosamente en Marte. Sin embargo, tras varias horas de espera por telemetría, la única señal recibida desde el planeta rojo fue el mensaje:

> “No pilot found”.

Para resolver este inconveniente, deberán implementar un agente inteligente capaz de aprender a navegar de forma autónoma en la superficie de Marte, aplicando técnicas de aprendizaje por refuerzo sobre un simulador.

El objetivo es que el agente eventualmente aprenda que avanzar suele ser mejor que no hacerlo.

---

## Proyecto MATE

### Martian Adversarial Tactics Engine

Una vez que el rover logre desplazarse sin mayores crisis existenciales, en **Red Destination™** se ha identificado una prioridad crítica para el éxito de la misión.

Diversos equipos de expertos han coincidido en que, ante un eventual contacto con vida inteligente en Marte, el desempeño del rover en dinámicas lúdicas tendrá el mayor impacto en futuras relaciones interplanetarias.

Por este motivo, resulta imperativo que el agente sea capaz de desenvolverse de manera competente en estos entornos.

Para ello, se utilizará un simulador basado en el juego de mesa **Isolation**, donde el agente deberá enfrentarse a un oponente en un escenario adversarial.

La tarea será implementar un algoritmo basado en **Minimax/Expectimax** que le permita tomar decisiones óptimas y minimizar el riesgo de resultados desfavorables.

---

# Tareas a desarrollar

## Proyecto LOST

La primera tarea está basada en el ambiente **Mountain Car Continuous**.

Concretamente, se pide:

### 1. Discretización de observaciones y acciones

Dado que las observaciones y acciones son continuas, deben discretizarse.

Se espera que exploren diferentes opciones, justificando:

- La elección realizada.
- El impacto en el agente.

### 2. Técnica

La técnica elegida para resolver el problema es **Q-Learning**.

### 3. Exploración de hiperparámetros

Se deberán explorar hiperparámetros para encontrar el algoritmo que obtenga mejores resultados.

Se espera que se experimenten múltiples combinaciones de hiperparámetros, justificando:

- La forma de evaluar el rendimiento del agente.
- La elección final de los hiperparámetros.

### 4. Componente de investigación

Se deberán leer los capítulos **8.1** y **8.2** del libro:

> *Reinforcement Learning: An Introduction*, de Sutton y Barto.

Además, se deberá implementar el algoritmo **Dyna-Q**.

Se espera que realicen un análisis y experimentación sobre el ambiente, de forma similar al trabajo con **Q-Learning**.

---

## Proyecto MATE

La segunda tarea está basada en el juego **Isolation**.

Concretamente, se pide:

### 1. Técnicas

Implementar tanto **Minimax** como **Expectimax** para decidir cuál es la mejor técnica para este caso.

En el caso de **Minimax**, debe implementarse utilizando **Alpha-Beta Pruning** y analizar su impacto.

### 2. Funciones de evaluación

Implementar funciones de evaluación que permitan analizar un estado dado.

Se espera que experimenten con las funciones, intentando distintas combinaciones y ponderaciones.

### 3. Experimentación

Definir pruebas para evaluar los agentes y hacer un registro completo de los resultados obtenidos.

---

# Auditoría

Para evaluar el desempeño de los agentes entrenados, deben entregar:

- Todo el código en Python:
  - Archivos `.py`
  - Archivos `.ipynb`
- Los modelos computados:
  - Archivos `.pkl`
  - Formatos similares
- Un informe de no más de 20 páginas, más anexos, en formato `.pdf`

Todo el contenido debe ser entregado en un archivo `.zip`.

Es obligatorio entregar al menos un modelo computado para el primer ejercicio. En caso contrario, el ejercicio será considerado como no hecho.

---

## Contenido del informe

El informe debe incluir:

### 1. Resumen del abordaje

Un resumen de cómo se abordó cada tarea, incluyendo información relevante como:

- Interacción con el simulador.
- Parámetros utilizados.
- Tiempo de ejecución.
- Resultados obtenidos.

### 2. Apoyo visual

Gráficos claros y comentarios que permitan entender el desempeño de las soluciones.

### 3. Notas de advertencia

Cualquier nota de advertencia que se desee comunicar.

Por ejemplo, en caso de haber encontrado dificultades, se deberá explicar:

- Cuáles fueron.
- Por qué no se pudieron solucionar.

---

## Criterios de evaluación

La evaluación se basará en la documentación entregada.

Es fundamental que el informe sea:

- Claro.
- Legible.
- Completo.
- Suficiente para comprender el enfoque, los resultados y las conclusiones del trabajo.

---

# Ambiente

Se utilizará **Poetry** para ambos ejercicios en entornos separados.

Se entregará código de ambos ambientes listo para ejecutar el simulador.

