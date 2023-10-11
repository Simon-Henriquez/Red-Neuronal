"""Implementación de conceptos de Inteligencia Artificial.

Construcción de Neuronas, Funciones de Activación, Función de
Agregación, Capas de Neuronas, Red Neuronal Completa y Procesos
de Propagación Hacia Adelante, Descendiente Gradiente, Propagación
Hacia Atras y Entrenamiento de una Red Neuronal.
"""

__version__ = "1.2"
__author__ = "Simón Henríquez" + " / " + "Felipe Mancilla"

import numpy as np
from numpy.typing import NDArray

from funciones_NN import (
    FuncionActivacion,
    Sigmoide,
    Relu,
    Tanh,
    Linear,
    FuncionAgregacion,
    Ponderacion,
    FuncionCoste,
    ErrorCuadraticoMedio
)

class CapaEntradaError(Exception):
    pass

class ParametrosDimensionesError(Exception):
    pass


class Neurona:
    def __init__(
            self,
            funcion_activacion: FuncionActivacion,
            funcion_agregacion: FuncionAgregacion):

        self.funcion_activacion = funcion_activacion
        self.funcion_agregacion = funcion_agregacion

    def acoplar_conexiones(self, cantidad_entradas: int):
        self.pesos = np.array([np.random.randn(cantidad_entradas)])
        self.bias = np.array([np.random.randn(1)])

    def calcular_salida(self, datos_entrada: NDArray) -> float:
        return self.funcion_activacion.calcular_resultado(datos_entrada)

    def calcular_agregacion(self, datos_entrada: NDArray) -> float:
        return self.funcion_agregacion.calcular_resultado(
            datos_entrada,
            self.pesos,
            self.bias
        )

    def __len__(self) -> int:
        try:
            return self.pesos.shape[1]
        except:
            return 0


class Capa:
    def __init__(
            self,
            cantidad_neuronas: int,
            funcion_agregacion: FuncionAgregacion = Ponderacion,
            funcion_activacion: FuncionActivacion = Linear):

        self.largo = cantidad_neuronas
        self.neuronas = [
            Neurona(funcion_activacion, funcion_agregacion)
            for _ in range(cantidad_neuronas)
        ]

    def calcular_salida_sin_gradiente(
            self,
            datos_entrada: NDArray) -> dict[str, NDArray]:

        salidas_capa = np.empty((0,1), np.float64)

        for neurona in self.neuronas:
            agregacion = neurona.calcular_agregacion(datos_entrada)
            salidas_capa = np.concatenate((
                salidas_capa,
                neurona.calcular_salida(agregacion)
            ), axis=0)

        return {"salidas_capa": salidas_capa}

    def calcular_salida_con_gradiente(
            self,
            datos_entrada: NDArray) -> dict[str, NDArray]:

        salidas_capa = np.empty((0,1), np.float64)
        parametros_capa = np.empty((0,datos_entrada.shape[0]+1), np.float64)
        derivadas_capa = np.empty((0,1), np.float64)
        derivadas_parametros = np.empty((0,datos_entrada.shape[0]+1), np.float64)
        for neurona in self.neuronas:
            parametros_neurona = np.concatenate(
                (neurona.pesos, neurona.bias),
                axis=1
            )
            parametros_capa = np.concatenate(
                (parametros_capa, parametros_neurona),
                axis=0
            )
            agregacion = neurona.calcular_agregacion(datos_entrada)
            salidas_capa = np.concatenate((
                salidas_capa,
                neurona.calcular_salida(agregacion)
            ), axis=0)
            derivadas_capa = np.concatenate((
                derivadas_capa,
                neurona.funcion_activacion.calcular_derivada(agregacion)
            ), axis=0)
            derivada_parametro = np.concatenate(
                (datos_entrada.T, np.array([[1]])),
                axis=1
            )
            derivadas_parametros = np.concatenate(
                (derivadas_parametros, derivada_parametro),
                axis=0
            )

        return {
            "salidas_capa": salidas_capa,
            "derivadas": [parametros_capa, derivadas_parametros, derivadas_capa]
        }

    def __len__(self) -> int:
        return len(self.neuronas)

    def __str__(self) -> str:
        return f"Soy una Capa con {self.largo} Neuronas."


class RedNeuronal:
    def __init__(self, *capas: Capa):
        self.capas = capas
        self._crear_conexiones()

    def propagacion_adelante(
            self,
            datos_entrada: NDArray,
            requiere_gradiente: bool = False) -> dict[str, NDArray]:

        if requiere_gradiente:
            return self._propagacion_adelante_con_gradiente(datos_entrada)
        return self._propagacion_adelante_sin_gradiente(datos_entrada)

    def _propagacion_adelante_sin_gradiente(
            self,
            datos_entrada: NDArray) -> dict[str, NDArray]:

        salida_capa_anterior = {"salidas_capa": datos_entrada}

        for capa in self.capas[1:]:
            salida_capa_actual = capa.calcular_salida_sin_gradiente(
                salida_capa_anterior["salidas_capa"]
            )
            salida_de_la_red = salida_capa_actual["salidas_capa"]
            salida_capa_anterior = salida_capa_actual

        return {"salida_red": salida_de_la_red}

    def _propagacion_adelante_con_gradiente(
            self,
            datos_entrada: NDArray) -> dict[str, NDArray]:

        salida_capa_anterior = {"salidas_capa": datos_entrada}
        derivadas_parametros = np.vstack((datos_entrada, np.array([[1]]))) #aaaaa
        salida_capa_anterior["derivadas_parametros"] = derivadas_parametros
        derivadas = []

        for capa in self.capas[1:]:
            salida_capa_actual = capa.calcular_salida_con_gradiente(
                salida_capa_anterior["salidas_capa"]
            )
            derivadas.extend(salida_capa_actual["derivadas"])
            salida_de_la_red = salida_capa_actual["salidas_capa"]
            salida_capa_anterior = salida_capa_actual

        return {"salida_red": salida_de_la_red, "derivadas": derivadas}

    def _crear_conexiones(self):
        for neurona in self.capas[0].neuronas:
            if not issubclass(neurona.funcion_activacion, Linear):
                raise CapaEntradaError(
                    "Las neuronas de la capa de entrada deben "
                    + "tener función de activación Linear"
                    )

        for index_anterior, capa in enumerate(self.capas[1:]):
            cantidad_neuronas_capa_anterior = self.capas[index_anterior].largo
            for neurona in capa.neuronas:
                neurona.acoplar_conexiones(cantidad_neuronas_capa_anterior)

    def printear_red(self):
        for index_capa, capa in enumerate(self.capas):
            print(f"Capa {index_capa}")
            for neurona in capa.neuronas:
                print(f"Neurona {neurona}")
                try:
                    print(neurona.pesos)
                    print(neurona.bias)
                except:
                    pass

    def __str__(self) -> str:
        mensaje = ""
        for capa in self.capas:
            mensaje += f"[{capa.largo},{len(capa.neuronas[0])}] ---> "
        return mensaje



class Entrenamiento:
    def __init__(
            self,
            datos_entrenamiento: NDArray,
            datos_validacion: NDArray,
            salidas_esperadas: NDArray,
            red_neuronal: RedNeuronal,
            epocas: int = 1,
            tasa_aprendizaje: float = 0.01,
            funcion_coste: FuncionCoste = ErrorCuadraticoMedio):

        self.epocas = epocas
        self.tasa_aprendizaje = tasa_aprendizaje
        self.red_neuronal = red_neuronal
        self.funcion_coste = funcion_coste
        self.datos_entrenamiento = datos_entrenamiento
        self.datos_validacion = datos_validacion
        self.salidas_esperadas = salidas_esperadas
        self.errores = []

    def entrenar_red(self):
        for ii in range(self.epocas):
            for index, datos in enumerate(self.datos_entrenamiento):
                datos = np.array([datos]).T
                gradientes = self._obtener_gradientes(datos, index)
                self._propagacion_hacia_atras(gradientes)

    def _propagacion_hacia_atras(self, derivadas: list[NDArray]):
        base = derivadas.pop(-1)
        for capa in self.red_neuronal.capas[:0:-1]:
            base = base * derivadas.pop(-1)
            self._actualizar_pesos(capa, derivadas=base * derivadas.pop(-1))
            base = base * derivadas.pop(-1)
            base = np.array(
                [[
                    sum(base[:,i])
                    for i in range(base.shape[1])
                ]]
            )
            base = np.delete(base, -1, 1)
            base = base.T

    def _obtener_gradientes(self, datos_entrada: NDArray, indice: int) -> list[NDArray]:
        salida_red = self.red_neuronal.propagacion_adelante(datos_entrada, requiere_gradiente=True)
        error = self.funcion_coste.calcular_coste(
            salida_red["salida_red"],
            np.array([self.salidas_esperadas[indice]]).T
        )
        self.errores.append(error)
        derivada_error = self.funcion_coste.calcular_derivada(
            salida_red["salida_red"],
            np.array([self.salidas_esperadas[indice]]).T
        )
        derivadas = salida_red["derivadas"]
        derivadas.append(derivada_error)
        return derivadas

    def _actualizar_pesos(self, capa: Capa, derivadas: NDArray):
        for index, neurona in enumerate(capa.neuronas):
            neurona.pesos = neurona.pesos - (self.tasa_aprendizaje * derivadas[index,:][:-1])
            neurona.bias = neurona.bias - (self.tasa_aprendizaje * derivadas[index,:][-1])


def red_neuronal_pesos_setter(
        matriz_parametros: list[NDArray],
        red_neuronal: RedNeuronal) -> RedNeuronal:

    mensaje_error = ("Matriz de parametros no "
                    +"concuerda con la arquitectura de la red.")
    if len(red_neuronal.capas)-1 != len(matriz_parametros):
        raise ParametrosDimensionesError(mensaje_error)
    for index_capa, capa in enumerate(red_neuronal.capas[1:]):
        if len(capa) != matriz_parametros[index_capa].shape[0]:
            raise ParametrosDimensionesError(mensaje_error)
        for index_neurona, neurona in enumerate(capa.neuronas):
            if len(neurona)+1 != matriz_parametros[index_capa][index_neurona].shape[0]:
                raise ParametrosDimensionesError(mensaje_error)
            parametros = np.concatenate((neurona.pesos, neurona.bias), axis=1)
            nuevos_pesos = np.array([matriz_parametros[index_capa][index_neurona]])
            if parametros[0].shape == nuevos_pesos[0].shape:
                neurona.pesos = np.delete(nuevos_pesos, -1, 1)
                neurona.bias = np.delete(nuevos_pesos, slice(0,-1), 1)


def example_app():
    # Datos para utilizar la red
    datos_para_predecir = np.array(
        [[0.05, 0.1]]
    )
    salidas_esperadas = np.array(
        [[0.01, 0.99]])
    salidas_validacion = salidas_esperadas

    # Creando Capas
    capa_entrada = Capa(2)
    capa_oculta1 = Capa(2, Ponderacion, Sigmoide)
    capa_salida = Capa(2, Ponderacion, Sigmoide)

    # Creando Red con Capas
    red_neuronal = RedNeuronal(
        capa_entrada,
        capa_oculta1,
        capa_salida
    )

    # Opcional
    # Solo por si quiere definir usted mismo los pesos iniciales.
    # De lo contrario comente las siguientes 3 líneas.
    pesos_capa1 = np.array(([0.15, 0.20, 0.35], [0.25, 0.30, 0.35]))
    pesos_capa2 = np.array(([0.40, 0.45, 0.6], [0.50, 0.55, 0.6]))
    red_neuronal_pesos_setter([pesos_capa1, pesos_capa2], red_neuronal)

    # Entrenando Red
    entrenamiento = Entrenamiento(
        datos_para_predecir,
        salidas_validacion,
        salidas_esperadas,
        red_neuronal,
        epocas=100,
        tasa_aprendizaje=0.5
    )
    entrenamiento.entrenar_red()

    # Usando la Red
    print("\n"+"#"*50)
    print("\nUsando la Red con Arquitectura:\n", red_neuronal)
    result = red_neuronal.propagacion_adelante(datos_para_predecir.T)
    print("\nResultado:\n", result["salida_red"])

if __name__ == '__main__':
    print("\nAutores: [" + __author__ + "]")
    print("Versión: " + __version__)
    example_app()