import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray

class FuncionActivacion(ABC):
    @staticmethod
    @abstractmethod
    def calcular_resultado(x: NDArray) -> NDArray:
        ...

    @staticmethod
    @abstractmethod
    def calcular_derivada(x: NDArray) -> NDArray:
        ...

    @abstractmethod
    def __str__(self) -> str:
        ...

class Sigmoide(FuncionActivacion):
    @staticmethod
    def calcular_resultado(x: NDArray):
        if x >= 0:
            return 1 / (1 + np.exp(-x))
        else:
            return np.exp(x) / (1 + np.exp(x))

    @staticmethod
    def calcular_derivada(x: NDArray) -> NDArray:
        salida = Sigmoide.calcular_resultado(x)
        result = salida * (1 - salida)
        return result

    def __str__(self) -> str:
        return f"Función de activación Sigmoide"

class Relu(FuncionActivacion):
    @staticmethod
    def calcular_resultado(x: NDArray) -> NDArray:
        return np.maximum(0, x)

    @staticmethod
    def calcular_derivada(x: NDArray) -> NDArray:
        return np.where(x > 0, 1, 0)

    def __str__(self) -> str:
        return f"Función de activación Relu"

class Tanh(FuncionActivacion):
    @staticmethod
    def calcular_resultado(x: NDArray) -> NDArray:
        # return np.where(x >= 0, (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x)), (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1))
        result = np.empty_like(x)
        for i in range(x.shape[0]):
            if x[i] >= 0:
                result[i] = (1 - np.exp(-2 * x[i])) / (1 + np.exp(-2 * x[i]))
            else:
                result[i] = (np.exp(2 * x[i]) - 1) / (np.exp(2 * x[i]) + 1)
        return result

    @staticmethod
    def calcular_derivada(x: NDArray) -> NDArray:
        return 1 - np.tanh(x)**2

    def __str__(self) -> str:
        return f"Función de activación Tangente hiperbólica"

class Linear(FuncionActivacion):
    @staticmethod
    def calcular_resultado(x: NDArray) -> NDArray:
        return x

    @staticmethod
    def calcular_derivada(x: NDArray) -> NDArray:
        return np.ones(x.shape)

    def __str__(self) -> str:
        return f"Función de activación Linear"

class SoftMax(FuncionActivacion):
    @staticmethod
    def calcular_resultado(x: NDArray) -> NDArray:
        numeradores = np.exp(x)
        return numeradores / np.sum(numeradores)

    @staticmethod
    def calcular_derivada(x: NDArray) -> NDArray:
        denominador = (np.sum(np.exp(x)))**2
        numeradores = []
        for index, num in enumerate(x):
            sin_actual = np.delete(x, index, 0)
            exponente = np.exp(sin_actual + num)
            suma = np.sum(exponente)
            numeradores.append(suma)
        return np.array([numeradores]).T / denominador


class FuncionAgregacion(ABC):
    @staticmethod
    @abstractmethod
    def calcular_resultado(
            entradas: NDArray,
            pesos: NDArray,
            bias: NDArray) -> NDArray:
        ...

class Ponderacion(FuncionAgregacion):
    @staticmethod
    def calcular_resultado(
            entradas: NDArray,
            pesos: NDArray,
            bias: NDArray) -> NDArray:
        return np.dot(pesos, entradas) + bias



class FuncionCoste(ABC):
    @staticmethod
    @abstractmethod
    def calcular_coste(predecido: NDArray, esperado: NDArray) -> NDArray:
        ...

    @staticmethod
    @abstractmethod
    def calcular_derivada(predecido: NDArray, esperado: NDArray) -> NDArray:
        ...

class ErrorCuadraticoMedio(FuncionCoste):
    @staticmethod
    def calcular_coste(predecido: NDArray, esperado: NDArray) -> NDArray:
        return (1/2) * ((predecido - esperado)**2)

    @staticmethod
    def calcular_derivada(predecido: NDArray, esperado: NDArray) -> NDArray:
        return predecido - esperado

class CrossEntropy(FuncionCoste):
    @staticmethod
    def calcular_coste(predecido: NDArray, esperado: NDArray) -> NDArray:
        pass
    @staticmethod
    def calcular_derivada(predecido: NDArray, esperado: NDArray) -> NDArray:
        pass