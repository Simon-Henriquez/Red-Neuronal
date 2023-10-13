import red_neuronal as NN

entradas = NN.np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
salidas = NN.np.array([[0.0], [1.0], [1.0], [0.0]])

capa_entrada = NN.Capa(2)
capa_oculta1 = NN.Capa(5, NN.Ponderacion, NN.Sigmoide) # Perceptron Multicapa
capa_salida = NN.Capa(1, NN.Ponderacion, NN.Linear)

red_neuronal = NN.RedNeuronal(
    capa_entrada,
    capa_oculta1,
    capa_salida
)

red_neuronal.printear_red()

entrenamiento = NN.Entrenamiento(
    entradas,
    salidas,
    salidas,
    red_neuronal,
    epocas=100,
    tasa_aprendizaje=0.5
)
entrenamiento.entrenar_red()

def printeando_resultados():
    print("\n"+"#"*50)

    print("\nUsando la Red con Arquitectura:\n", red_neuronal)

    result = red_neuronal.propagacion_adelante(entradas[0].T)
    print(f"\nResultado: {entradas[0].T}\n", result["salida_red"])

    result = red_neuronal.propagacion_adelante(entradas[1].T)
    print(f"\nResultado: {entradas[1].T}\n", result["salida_red"])

    result = red_neuronal.propagacion_adelante(entradas[2].T)
    print(f"\nResultado: {entradas[2].T}\n", result["salida_red"])

    result = red_neuronal.propagacion_adelante(entradas[3].T)
    print(f"\nResultado: {entradas[3].T}\n", result["salida_red"])

printeando_resultados()

entrenamiento.perdida_vs_epoca()