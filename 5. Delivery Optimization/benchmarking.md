# Benchmarks: Genetic Algorithms, Simulated Annealing & Reinforcement Learning

Definir los tests que se deben ejecutar y que variables medir.

## Reducción de Datos
En la carpeta "02. Datos" se han guardado 4 reducciones de datos:

| Nombre 	| Pedidos 	| Camiones 	|
|--------	|---------	|----------	|
| tiny2  	| 50      	| 10       	|
| small2 	| 100     	| 20       	|
| med2   	| 500     	| 35       	|
| large2 	| 750     	| 50       	|

Estas 4 reducciones formarán parte de los benchmarks iniciales entre los algoritmos a estudiar.

## Casos de Experimentación
1. Tiempo fijo: Se comparan los resultados obtenidos al cabo de ejecutar cada reducción durante 15 minutos (900 segundos).

2. Mejor coste: Se ejecuta hasta intentar que el algoritmo converja y llegue a una solución mínima. Comparar los tiempos y los costes de cada uno.



## Valores a devolver

- Coste de la solución encontrada en kilómetros. Solo pasar los kilometros que se recorren en vacío.
- Tiempo de ejecución. Medidos en segundos.
- Número de pedidos que se entregan tarde. (Datos extra: media de cuantos días más tarde lo entrega).

## Formato
Fichero csv nombrado : [nombreAlgortimo]_benchmark_reduced.csv

El csv debe contener las siguientes columnas rellenas para cada reducción del dataset y para cada caso que se experimenta.

| dataset 	| km       	| tiempo 	| pedidosTardes 	| experimento 	|
|---------	|----------	|--------	|---------------	|-------------	|
| tiny2   	| 12000.00 	| 900.00 	| 0             	| tiempo_fijo 	|
| tiny2   	| 12000.00 	| 400.00 	| 0             	| mejor_coste 	|


***

# Benchmarks: Genetic Algorithms, Simulated Annealing & Pseudo Constraint Programming

## Datos
El fichero con el dataset base completo + la replanificación 24 (2022-08-13 08:00).

## Casos de Experimentación.
1. Planificación base
2. Planificación base + replanificación


## Valores a devolver

- Coste de la solución encontrada en kilómetros. Solo pasar los kilometros que se recorren en vacío.
- Tiempo de ejecución. Medidos en segundos.
- Número de pedidos que se entregan tarde. (Datos extra: media de cuantos días más tarde lo entrega).

## Valores a devolver

- Coste de la solución encontrada en kilómetros. Solo pasar los kilometros que se recorren en vacío.
- Tiempo de ejecución. Medidos en segundos.
- Número de pedidos que se entregan tarde. (Datos extra: media de cuantos días más tarde lo entrega).

## Formato
Fichero csv nombrado : [nombreAlgortimo]_benchmark_total.csv

El csv debe contener las siguientes columnas rellenas para cada reducción del dataset y para cada caso que se experimenta.

| km       	| tiempo 	| pedidosTardes 	| experimento 	    |
|----------	|--------	|---------------	|-------------	    |
| 12000.00 	| 900.00 	| 0             	| base 	            |
| 12000.00 	| 400.00 	| 0             	| replanificacion 	|

