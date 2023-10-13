# Redes Neuronales de Scratch
> Atención este programa tiene muchos ejercicios de Algebra Linear. No nos hacemos responsables por los **daños** que le pueda llegar a causar la lectura del código.
Creación de *Redes Neuronales* usando la librería numpy.
Programa realizado para el Laboratorio de Inteligencia Artificial con el profesor **Cristián Inzulza Castro**.

**Desarrolladores**
- Felipe Mancilla
- Simón Henríquez
- Hugo Concha

# Cómo Usar el Código
## Modulo funciones_NN.py
En este package están las funciones de `Agregación`, `Activación` y `Coste` que estan habilitadas por ahora para implementarlas en la red neuronal.
## Modulo red_neuronal.py
Este es el programa principal, aquí están las clases `Neurona`, `Capa`, `RedNeuronal`, `Entrenamiento`.
> Si usas pipenv para tu entorno virtual aquí tienes un commando que puede ser de tu ayuda.
> ```
> pipenv run python src/red_neuronal.py
> ```
Para crear una Red Neuronal puede guiarse con la funcion ejemplo `example_app`.
Si le quedan dudas de como funciona una red neuronal vea el siguiente tutorial que profundiza mas en el tema:
1. Crear 2 matrices fila, una para los datos de entrada a la red y otra para las salidas esperadas de la red. Si quisiera una red de 3 entradas y 2 salidas:
	```python
	datos_entrada = np.array([[0.5, 0.9, 0.2]])
	datos_salida_esperada = np.array([[0.9, 0.3]])
	```
2. Crear las Capas de la Red Neuronal. Usted puede crear cuantas capas quiera, asi mismo, definir la cantidad de Neuronas que guste para cada Capa. Por ejemplo, 3 Capas ocultas, la primera con 4 Neuronas y función de Activacón Tangente Hiperbólica, la segunda 3 Neuronas y Relu y la última 5 Neuronas y Sigmoide.
	```python
	capa_entrada = Capa(2)
	capa_oculta1 = Capa(4, Ponderacion, Tanh)
	capa_oculta2 = Capa(3, Ponderacion, Relu)
	capa_oculta3 = Capa(5, Ponderacion, Sigmoide)
	capa_salida = Capa(2, Ponderacion, Sigmoide)
	```
	>  **Nota**: Fíjese que la capa de entrada y la de salida deben coincidir con los datos definidos en el **Paso 1**. Seguramente también notó que la capa de entrada no tiene función de Activación, esto es porque la capa de entrada siempre será Linear sin importar lo que se defina.
3. Luego de tener las Capas creadas hay que crear la Red Neuronal:
	```python
	red_neuronal = RedNeuronal(
		capa_entrada,
		capa_oculta1,
		capa_oculta2,
		capa_oculta3,
		capa_salida
	)
	```
4. Para entrenar la red debe crear un objeto Entrenamiento pasandole almenos cinco argumentos.
	```python
	entrenamiento = Entrenamiento(
		entradas_entrenamiento,
		salidas_entrenamiento,
		entradas_validacion,
		salidas_validacion,
		red_neuronal
	)
	```
	> Por lo general los datos de validacion en tamaño son como un 10% de los datos de entrenamiento.
	Luego para que comienze el entrenamiento:
	```python
	entrenamiento.entrenar_red()
	```
6. Para probar el funcionamiento de la nueva red entrenada:
	```python
	resultado = red_neuronal.propagacion_adelante(entrada_testeo)

	print(resultado["salida_red"])
	```
## Jupyter Notebook imagen.ipynb
Este notebook tiene un ejemplo de como se ve una imágen de los números escritos a mano.

## Jupyter Notebook mnist.ipynb
Este notebook también es un ejemplo de mayor complejidad en el que se crea una red neuronal capaz de predecir el número escrito a mano que se encuentra en una imágen de 28 x 28 pixeles.

## data.py
Este modulo tiene funciones que son usadas para obtener los datos de los csv para entrenar, validar y testear una red de reconocimiento de imagenes escritas a mano en el jupyter notebook *mnist.ipynb*.

## Como ejecutar los test
Para ejecutar los test primero debe ubicarse en el directorio src.
```
$ cd src
$ pipenv run python -m unittest tests/integration/test_red_neuronal.py
```