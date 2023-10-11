import unittest
import numpy as np

from red_neuronal import (
    Capa, RedNeuronal,
    Entrenamiento,
    red_neuronal_pesos_setter
)
from funciones_NN import (
    Sigmoide,
    Ponderacion,
)

#pipenv shell
#pipenv run python -m unittest tests/integration/test_red_neuronal.py     

class TestRedNeuronal(unittest.TestCase):
    def setUp(self):
        # Datos para utilizar la red
        self.datos_para_predecir = np.array([[0.05, 0.1]])
        self.salidas_esperadas = np.array([[0.01, 0.99]])
        self.datos_validacion = self.datos_para_predecir

        # Creando Capas
        capa_entrada = Capa(2)
        capa_oculta1 = Capa(2, Ponderacion, Sigmoide)
        capa_salida = Capa(2, Ponderacion, Sigmoide)

        # Creando Red con Capas
        self.red_neuronal = RedNeuronal(
            capa_entrada,
            capa_oculta1,
            capa_salida
        )

        # Opcional
        # Solo por si quiere definir usted mismo los pesos iniciales.
        # De lo contrario comente las siguientes 3 líneas.
        pesos_capa1 = np.array(([0.15, 0.20, 0.35], [0.25, 0.30, 0.35]))
        pesos_capa2 = np.array(([0.40, 0.45, 0.60], [0.50, 0.55, 0.60]))
        red_neuronal_pesos_setter([pesos_capa1, pesos_capa2], self.red_neuronal)

    def test_red_neuronal_output(self):
        result = self.red_neuronal.propagacion_adelante(
            np.array([self.datos_para_predecir[0]]).T
        )
        expected =  np.array([[0.75136507], [0.77292846]])
        self.assertTrue(np.allclose(result["salida_red"], expected, atol=0.0001))

    def test_red_neuronal_back_propagation(self):
        entrenamiento = Entrenamiento(
            self.datos_para_predecir,
            self.datos_validacion,
            self.salidas_esperadas,
            self.red_neuronal,
            epocas=100,
            tasa_aprendizaje=0.5
        )
        entrenamiento.entrenar_red()
        result = self.red_neuronal.propagacion_adelante(
            np.array([self.datos_para_predecir[0]]).T
        )

        e = np.array([[0.05], [0.1]])
        p1 = np.array([[0.15, 0.20], [0.25, 0.30]])
        b1 = np.array([[0.35], [0.35]])
        vh = np.dot(p1,e) + b1
        h = 1/(1+np.exp(-vh))

        p2 = np.array([[0.4, 0.45], [0.50, 0.55]])
        b2 = np.array([[0.6], [0.6]])
        vo = np.dot(p2, h) + b2
        o = 1/(1+np.exp(-vo))
        o_ = np.array([[0.01], [0.99]])
        error = (1/2) * (o - o_)**2
        for x in range(100):
            vh = np.dot(p1,e) + b1
            h = 1/(1+np.exp(-vh))

            vo = np.dot(p2, h) + b2
            o = 1/(1+np.exp(-vo))
            o_ = np.array([[0.01], [0.99]])
            error = (1/2) * (o - o_)**2


            # Derivada del error con respecto a O
            de_do = o - o_
            #Derivada del O con respecto a su entrada
            do_dvo = o * (1-o)
            a = np.concatenate((h.T, h.T), axis=0)
            # Derivada de la entrada vO con respecto a los pesos
            bbbb = np.array([[1], [1]])
            dvo_dvp2 = np.concatenate((a, bbbb), axis=1)
            # Gradientes P2 `actualizar`
            dE_dp2 = de_do * do_dvo * dvo_dvp2

            # Derivada de la entrada vO con respecto a las neuronas anteriores
            dvo_dh = p2
            # Derivada del error con respecto a H
            a = de_do * do_dvo * dvo_dh
            dE_dh = np.array([np.sum(a, axis=0)]).T
            # Derivada de capa1 con su entrada
            dH_dvh = h * (1-h)
            #Derivada VH con respecto a los pesos
            a = np.concatenate((e.T, e.T), axis=0)
            bbb = np.array([[1], [1]])
            dvh_dp1 = np.concatenate((a, bbb), axis=1)
            #Gradiente P1 `actualizar p1`
            dE_dp1 = dE_dh * dH_dvh * dvh_dp1

            # actualizar pesos2
            derivada_tasa = 0.5 * dE_dp2
            p2 = p2 - derivada_tasa[:,:-1]
            b2 = np.array(b2 - np.array([derivada_tasa[:,-1]]).T)

            # actualizar pesos1
            derivada_tasa = 0.5 * dE_dp1
            p1 = p1 - derivada_tasa[:,:-1]
            b1 = np.array(b1 - np.array([derivada_tasa[:,-1]]).T)


        vh = np.dot(p1,e) + b1
        h = 1/(1+np.exp(-vh))

        vo = np.dot(p2, h) + b2
        expected = 1/(1+np.exp(-vo))

        self.assertTrue(np.allclose(result["salida_red"], expected, atol=0.0001))