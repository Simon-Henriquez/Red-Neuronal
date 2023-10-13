import numpy as np
from abc import ABC, abstractmethod
from numpy.typing import NDArray

class FuncionActivacion(ABC):
    @staticmethod
    @abstractmethod
    def adelante(x: NDArray) -> NDArray:
        ...

    @staticmethod
    @abstractmethod
    def atras(x: NDArray) -> NDArray:
        ...

    @staticmethod
    @abstractmethod
    def __str__() -> str:
        ...

class Sigmoide(FuncionActivacion):
    @staticmethod
    def adelante(x: NDArray):
        if x >= 0:
            return 1 / (1 + np.exp(-x))
        else:
            return np.exp(x) / (1 + np.exp(x))

    @staticmethod
    def atras(x: NDArray) -> NDArray:
        salida = Sigmoide.adelante(x)
        result = salida * (1 - salida)
        return result

    @staticmethod
    def __str__() -> str:
        return f"sigmoide"

class Relu(FuncionActivacion):
    @staticmethod
    def adelante(x: NDArray) -> NDArray:
        return np.maximum(0, x)

    @staticmethod
    def atras(x: NDArray) -> NDArray:
        return np.where(x > 0, 1, 0)

    @staticmethod
    def __str__() -> str:
        return f"relu"

class Tanh(FuncionActivacion):
    @staticmethod
    def adelante(x: NDArray) -> NDArray:
        # return np.where(x >= 0, (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x)), (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1))
        result = np.empty_like(x)
        for i in range(x.shape[0]):
            if x[i] >= 0:
                result[i] = (1 - np.exp(-2 * x[i])) / (1 + np.exp(-2 * x[i]))
            else:
                result[i] = (np.exp(2 * x[i]) - 1) / (np.exp(2 * x[i]) + 1)
        return result

    @staticmethod
    def atras(x: NDArray) -> NDArray:
        return 1 - np.tanh(x)**2

    @staticmethod
    def __str__() -> str:
        return f"tanh"

class Linear(FuncionActivacion):
    @staticmethod
    def adelante(x: NDArray) -> NDArray:
        return x

    @staticmethod
    def atras(x: NDArray) -> NDArray:
        return np.ones(x.shape)

    @staticmethod
    def __str__() -> str:
        return f"linear"



class FuncionAgregacion(ABC):
    @staticmethod
    @abstractmethod
    def adelante(
            entradas: NDArray,
            pesos: NDArray,
            bias: NDArray) -> NDArray:
        ...

class Ponderacion(FuncionAgregacion):
    @staticmethod
    def adelante(
            entradas: NDArray,
            pesos: NDArray,
            bias: NDArray) -> NDArray:
        return np.dot(pesos, entradas) + bias



class FuncionCoste(ABC):
    @staticmethod
    @abstractmethod
    def adelante(predecido: NDArray, esperado: NDArray) -> NDArray:
        ...

    @staticmethod
    @abstractmethod
    def atras(predecido: NDArray, esperado: NDArray) -> NDArray:
        ...

class ErrorCuadraticoMedio(FuncionCoste):
    @staticmethod
    def adelante(predecido: NDArray, esperado: NDArray) -> NDArray:
        return (1/2) * ((predecido - esperado)**2)

    @staticmethod
    def atras(predecido: NDArray, esperado: NDArray) -> NDArray:
        return predecido - esperado