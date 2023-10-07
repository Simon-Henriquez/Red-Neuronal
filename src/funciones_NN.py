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
    def calcular_resultado(x: NDArray) -> NDArray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def calcular_derivada(x: NDArray) -> NDArray:
        return Sigmoide.calcular_resultado(x) * (1 - Sigmoide.calcular_resultado(x))

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
        return np.tanh(x)

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
        return 1

    def __str__(self) -> str:
        return f"Función de activación Linear"



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