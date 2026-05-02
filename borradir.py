import numpy as np
from medmnist import FractureMNIST3D
import multiprocessing as mp
import random

from tensorflow.python.ops.numpy_ops.np_dtypes import uint8


def Nivelar(n_p, num_pasos, matriz_datos):
    if n_p > num_pasos: raise ValueError(f"n_p ({n_p}) no puede superar num_pasos ({num_pasos})")
    s = num_pasos % n_p    # residuo 
    t = num_pasos // n_p   # capas de cada procesador
    out = []
    punto_actual = 0       
    
    for i in range(n_p):
        # Si hay residuo (s > 0), le damos 1 capa extra a este núcleo
        pasos_locales = t + 1 if s > 0 else t
        inicio_bloque = punto_actual
        #solo cambiamos la z (profundidad)
        bloque_asignado = matriz_datos[:, inicio_bloque:punto_actual + pasos_locales, :, :]
        out.append((inicio_bloque, bloque_asignado))
        # el siguiente empezará donde termina el actual
        punto_actual = punto_actual + pasos_locales
        if s>0 : s -= 1
        
    return out
#------------ Calculamos la entropía de cada capa en el dataset --------------------
def Entropia(tupla_datos):
    inicio , matriz =tupla_datos
    #aquí guardamos resultados
    r =[]
    #contamos las capas z q nos mandaron
    capas = matriz.shape[1]
    
    for i in range(capas):
        indice = inicio + i
        pixeles = matriz[:, i, :, :].flatten().astype(uint8)
        # frecuencias de intensidad (0 a 255)
        conteos = np.bincount(pixeles, minlength=256)
        #probabilidad (el número de casos donde aparece sobre el numero de casos totales)
        p = conteos / np.sum(conteos)
        # solo tomamos las probabilidades distintas de 0
        p = p[p>0]
        # Entropia: - sum(p * log2(p))
        entropia = -np.sum(p * np.log2(p))
        r.append((indice, entropia))
    
    return r

def Metricas(tupla_datos):
    inicio, matriz = tupla_datos
    r = []
    capas = matriz.shape[1]

    for i in range(capas):
        indice = inicio + i
        capa = matriz[:, i, :, :]  # (N, Y, X)

        # entropía por imagen individual, guardadas en lista
        entropias_por_imagen = []
        for n in range(capa.shape[0]):
            pixeles = capa[n].flatten().astype(np.uint8)
            conteos = np.bincount(pixeles, minlength=256)
            p = conteos / np.sum(conteos)
            p = p[p > 0]
            entropias_por_imagen.append(-np.sum(p * np.log2(p)))

        entropias_por_imagen = np.array(entropias_por_imagen)

        # promedio y std sobre las 1027 imágenes
        entropia_media = entropias_por_imagen.mean()
        entropia_std   = entropias_por_imagen.std()

        r.append((indice, entropia_media, entropia_std))

    return r


def init_centroides(datos, cant_centroides):
    centroides = []
    indices = random.sample(range(len(datos)), cant_centroides)
    for i in indices:
        centroides.append(list(datos[i]))
    return centroides

#función para asignar a cada pixel un cluster
def asignacion_clusters(datos, cant_centroides, centroides):
    asignaciones = [0] * len(datos)
    for i in range(len(datos)):
        dist_minima = float('inf')
        cluster_cercano = 0
        for j in range(cant_centroides):
            # Solo existe un valor: datos[i][0]
            diff = datos[i][0] - centroides[j][0]
            dist = abs(diff) # Distancia unidimensional
            
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
            promedio = [suma/contador]
            nuevos_centroides.append(promedio)
        else:
            nuevos_centroides.append(centroides_anteriores[j])

    return nuevos_centroides

#ahora una función para saber realmente cuanto se movieron los centroides
def movimiento_centroides(viejos_centroides, nuevos_centroides, cant_centroides):
    cambio_total = 0
    for i in range(cant_centroides):
        cambio = viejos_centroides[i][0] - nuevos_centroides[i][0]
        cambio_total += abs(cambio)
    return cambio_total

#función principal :)
def ejecutar_kmeans_lineal(datos, K, tol):
    centroides = init_centroides(datos, K)
    while True:
        etiquetas = asignacion_clusters(datos, K, centroides)
        # Usamos [c[:] for c in centroides] para asegurar una copia real de los valores
        centroides_anteriores = [c[:] for c in centroides]
        centroides = actualizar_centroides(datos, etiquetas, K, centroides_anteriores)
        cambio = movimiento_centroides(centroides_anteriores, centroides, K)
        if cambio < tol:
            print(f"¡Convergencia alcanzada!")
            break
    return centroides, etiquetas

if __name__ == '__main__':
    #cargamos el dataset
    dataset = FractureMNIST3D(split="train", download=True)
    #numero de procesadores
    n_p =3 
    
    #guardamos solo las imagenes
    volumenes = dataset.imgs
    #dimensiones de cada cosa
    N, Z, Y, X = dataset.imgs.shape
    
    print(f" {N} Imagenes con {Z} capas.")

    tareas = Nivelar(n_p, Z, volumenes)
    print(f"Pool con {n_p} trabajadores")

    with mp.Pool(processes=n_p) as pool:
        resultados = pool.map(Metricas, tareas)

    metricas_finales = []
    for lista_local in resultados:
        metricas_finales.extend(lista_local)

    metricas_finales.sort(key=lambda x: x[0])

    print(f"\n{'Capa':<8} {'Media':>10} {'Std':>10}")
    print("-" * 32)
    for indice, media, std in metricas_finales:
        print(f"Z={indice:02d}   {media:>10.4f} {std:>10.4f}")

    # K = 2
    # margen_error = 0.001
    # centroides_finales, etiquetas_finales = ejecutar_kmeans_lineal(valores_entropia, K, margen_error)
    #
    # print(f"Centroide 1 (Posible ruido): {centroides_finales[0][0]:.4f}")
    # print(f"Centroide 2 (Posible tejido): {centroides_finales[1][0]:.4f}")
    #
    # umbral = (centroides_finales[0][0] + centroides_finales[1][0]) / 2
    # print(f"Umbral calculado: {umbral:.4f}")
    # print (f"etiquetas por capa: {etiquetas_finales}")
