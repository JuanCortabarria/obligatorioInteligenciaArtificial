## Juego Isolation 

Isolation es un juego de tablero para dos jugadores, diseñado en 1972. El objetivo principal consiste en restringir progresivamente el espacio de movimiento del oponente hasta dejarlo sin movimientos legales. 

El juego se desarrolla sobre un tablero rectangular de dimensiones variables (en nuestro entorno, por defecto de 4 × 4). 

## Componentes y reglas 

El tablero contiene dos fichas, una por jugador, casillas transitables y casillas eliminadas o bloqueadas. 

Al comienzo de la partida, cada jugador comienza en una posición arbitraria. Todas las demás casillas permanecen disponibles para su uso. 

Los jugadores alternan turnos. En cada turno, un jugador debe realizar dos acciones: 

- Mover su ficha exactamente una casilla en cualquiera de las ocho direcciones posibles: 

   - Arriba. 

   - Abajo. 

   - Izquierda. 

   - Derecha. 

   - Diagonales. 

- Eliminar una casilla libre del tablero, pudiendo seleccionar cualquier posición que no esté ocupada por alguno de los jugadores ni haya sido previamente eliminada. 

Una vez eliminada, esa casilla queda inaccesible durante toda la partida. 

La **condición de victoria** se alcanza cuando un jugador logra dejar al oponente sin movimientos legales disponibles. En ese caso, la utilidad terminal es +1 si el agente gana y -1 si el agente pierde. 

## Espacio de observaciones y acciones 

El **espacio de observaciones** está compuesto por el estado completo del tablero. Cada vez que se ejecuta env.step(...), el entorno devuelve una observación del estado resultante. En la implementación, esta observación es un objeto de tipo Board, que permite consultar la posición de ambos jugadores, las casillas bloqueadas y las acciones disponibles (puede ver los detalles en board.py). 

El agente puede observar: 

- La disposición completa del tablero. 

- La ubicación de ambas fichas. 

- Las casillas bloqueadas. 

Esto convierte al juego en un entorno completamente observable. 

El **espacio de acciones** puede representarse como una combinación de dos decisiones, el movimiento del jugador y la posición de la casilla a eliminar. 

De esta manera, el conjunto de acciones posibles en un estado (s) se define como: 

## A(s) =  M(s) × R(s) 

donde: 

- M(s) es el conjunto de movimientos legales. 

- R(s) es el conjunto de casillas removibles. 

Por lo tanto es importante aplicar alfa-beta pruning para acelerar tanto Minimax como Expectimax podando ramas que no aportan información. 

Desde la perspectiva de características del entorno en el contexto de Inteligencia Artificial, Isolation presenta las siguientes características: 

- **Completamente observable** : ambos jugadores tienen acceso a toda la información del tablero. 

- **Determinista** : una acción siempre produce el mismo resultado dado un estado. 

- **Secuencial** : cada acción afecta directamente las futuras decisiones posibles. 

- **De suma cero** : la ganancia de un jugador equivale exactamente a la pérdida del otro. 

