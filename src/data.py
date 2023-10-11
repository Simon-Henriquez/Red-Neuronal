import csv
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def datos_entrenamiento() -> tuple[NDArray]:
    entradas_train = []
    salidas_train = []

    with open("data/datos_entrenamiento.csv", mode='r') as archivo_csv:
        lector_csv = csv.reader(archivo_csv)
        next(lector_csv)

        for fila in lector_csv:
            lista = eval(fila[0])
            numero = int(fila[1])
            entradas_train.append(lista)
            salidas_train.append([numero])
    return (np.array(entradas_train), np.array(salidas_train))


def datos_validacion() -> tuple[NDArray]:
    entradas_validacion = []
    salidas_validacion = []

    with open("data/datos_validacion.csv", mode='r') as archivo_csv:
        lector_csv = csv.reader(archivo_csv)
        next(lector_csv)

        for fila in lector_csv:
            lista = eval(fila[0])
            numero = int(fila[1])
            entradas_validacion.append(lista)
            salidas_validacion.append([numero])
    return (np.array(entradas_validacion), np.array(salidas_validacion))

def datos_testeo() -> tuple[NDArray]:
    entradas_test = []
    salidas_test = []

    with open("data/datos_testeo.csv", mode='r') as archivo_csv:
        lector_csv = csv.reader(archivo_csv)
        next(lector_csv)

        for fila in lector_csv:
            lista = eval(fila[0])
            numero = int(fila[1])
            entradas_test.append(lista)
            salidas_test.append([numero])
    return (np.array(entradas_test), np.array(salidas_test))


def dibujar_una_imagen(
        entradas: NDArray,
        salida_esperada: float,
        salida_predecida: float):

    imagen = entradas.reshape([28,28])
    plt.title(f"Esperada: {salida_esperada}, Predecida: {salida_predecida}")
    plt.imshow(imagen, cmap=plt.get_cmap('gray_r'))
    plt.show()

if __name__ == "__main__":
    print(len(datos_entrenamiento()))
    print(len(datos_validacion()))
    print(len(datos_testeo()))