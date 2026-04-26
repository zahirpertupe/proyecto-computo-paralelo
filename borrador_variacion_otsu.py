import numpy as np
from medmnist import FractureMNIST3D
import multiprocessing as mp
import random

from sympy.codegen.ast import continue_
from tensorflow.python.ops.numpy_ops.np_dtypes import uint8


def Nivelar(n_p, num_pasos, matriz_datos):
    if n_p > num_pasos: raise ValueError(f"n_p ({n_p}) no puede superar num_pasos ({num_pasos})")
    s = num_pasos % n_p  # residuo
    t = num_pasos // n_p  # capas de cada procesador
    out = []
    punto_actual = 0

    for i in range(n_p):
        # Si hay residuo (s > 0), le damos 1 capa extra a este núcleo
        pasos_locales = t + 1 if s > 0 else t
        inicio_bloque = punto_actual
        # solo cambiamos la z (profundidad)
        bloque_asignado = matriz_datos[:, inicio_bloque:punto_actual + pasos_locales, :, :]
        out.append((inicio_bloque, bloque_asignado))
        # el siguiente empezará donde termina el actual
        punto_actual = punto_actual + pasos_locales
        if s > 0: s -= 1

    return out


# ------------ Calculamos la entropía de cada capa en el dataset --------------------
def Entropia(tupla_datos):
    inicio, matriz = tupla_datos
    # aquí guardamos resultados
    r = []
    # contamos las capas z q nos mandaron
    capas = matriz.shape[1]

    for i in range(capas):
        indice = inicio + i
        pixeles = matriz[:, i, :, :].flatten().astype(uint8)
        # frecuencias de intensidad (0 a 255)
        conteos = np.bincount(pixeles, minlength=256)
        # probabilidad (el número de casos donde aparece sobre el numero de casos totales)
        p = conteos / np.sum(conteos)
        # solo tomamos las probabilidades distintas de 0
        p = p[p > 0]
        # Entropia: - sum(p * log2(p))
        entropia = -np.sum(p * np.log2(p))
        r.append((indice, entropia))

    return r


def init_centroides(datos, cant_centroides):
    centroides = []
    indices = random.sample(range(len(datos)), cant_centroides)
    for i in indices:
        centroides.append(list(datos[i]))
    return centroides


# función para asignar a cada pixel un cluster
def asignacion_clusters(datos, cant_centroides, centroides):
    asignaciones = [0] * len(datos)
    for i in range(len(datos)):
        dist_minima = float('inf')
        cluster_cercano = 0
        for j in range(cant_centroides):
            # Solo existe un valor: datos[i][0]
            diff = datos[i][0] - centroides[j][0]
            dist = abs(diff)  # Distancia unidimensional

            if dist < dist_minima:
                dist_minima = dist
                cluster_cercano = j

        asignaciones[i] = cluster_cercano
    return asignaciones


def actualizar_centroides(datos, asignaciones, cant_centroides, centroides_anteriores):
    nuevos_centroides = []
    for j in range(cant_centroides):
        suma = 0
        contador = 0
        for i in range(len(datos)):
            if asignaciones[i] == j:
                suma += datos[i][0]
                contador += 1

        if contador > 0:
            promedio = [suma / contador]
            nuevos_centroides.append(promedio)
        else:
            nuevos_centroides.append(centroides_anteriores[j])

    return nuevos_centroides


# ahora una función para saber realmente cuanto se movieron los centroides
def movimiento_centroides(viejos_centroides, nuevos_centroides, cant_centroides):
    cambio_total = 0
    for i in range(cant_centroides):
        cambio = viejos_centroides[i][0] - nuevos_centroides[i][0]
        cambio_total += abs(cambio)
    return cambio_total


def umbral_otsu(valores):
    #los valores son mi array de una dimensión con mis entropías
    v = np.array(valores).flatten()
    # discretizamos en N bins para buscar el mejor corte
    N = 1000
    mn, mx = v.min(), v.max()
    candidatos = np.linspace(mn, mx, N)  #obtenemos mil valores uniformemente espacioados entre el min y max de las entropías

    mejor_varianza = -1
    mejor_umbral = candidatos[0]

    for t in candidatos:
        clase0 = v[v <= t]
        clase1 = v[v > t]     #vamos iterando haciendo una partición de las clases con los mil valores (creo q se puede paralelizar esto también (? )
        if len(clase0) == 0 or len(clase1) == 0:
            continue

        w0 = len(clase0) / len(v)
        w1 = len(clase1) / len(v)  #proporción de clases

        #varianza inter-clase
        varianza_inter = w0 * w1 * (clase0.mean() - clase1.mean()) ** 2

        if varianza_inter > mejor_varianza:
            mejor_varianza = varianza_inter
            mejor_umbral = t  #actualizamos a aquel que produjo la mayor separación entre grupos

    return mejor_umbral


if __name__ == '__main__':
    # cargamos el dataset
    dataset = FractureMNIST3D(split="train", download=True)
    # numero de procesadores
    n_p = 3

    # guardamos solo las imagenes
    volumenes = dataset.imgs
    # dimensiones de cada cosa
    N, Z, Y, X = dataset.imgs.shape

    print(f" {N} Imagenes con {Z} capas.")

    tareas = Nivelar(n_p, Z, volumenes)
    print(f"Pool con {n_p} trabajadores")

    # paralelización
    with mp.Pool(processes=n_p) as pool:
        resultados = pool.map(Entropia, tareas)

    # ordenamos por indice
    entropias_finales = []
    for lista_local in resultados:
        entropias_finales.extend(lista_local)

    print("\nResultados de entropía por capa")
    for indice, entropia in entropias_finales:
        print(f"Capa Z={indice:02d} | Entropía: {entropia:.4f}")
    valores_entropia = np.array([item[1] for item in entropias_finales]).reshape(-1, 1)

    #print(valores_entropia)
    umbral = umbral_otsu(valores_entropia)
    print(f"Umbral Otsu: {umbral:.4f}")
    capas_utiles = 0
    for indice, entropia in entropias_finales:
        #print(entropia)
        if entropia > umbral:
            capas_utiles += 1

    print(f"Capas Utiles: {capas_utiles}")
