import numpy as np
from medmnist import FractureMNIST3D
import multiprocessing as mp
import os
import time

def Nivelar(n_p, num_pasos, matriz_datos):
    if n_p > num_pasos: raise ValueError(f"n_p ({n_p}) no puede superar num_pasos ({num_pasos})")
    s = num_pasos % n_p    # residuo
    t = num_pasos // n_p   # capas de cada procesador
    out = []
    punto_actual = 0

    for i in range(n_p):
        #si hay residuo (s > 0), le damos 1 capa extra a este núcleo
        pasos_locales = t + 1 if s > 0 else t
        inicio_bloque = punto_actual
        #solo cambiamos la z (profundidad)
        bloque_asignado = matriz_datos[:, inicio_bloque:punto_actual + pasos_locales, :, :]
        out.append((inicio_bloque, bloque_asignado))
        # el siguiente empezará donde termina el actual
        punto_actual = punto_actual + pasos_locales
        if s>0 : s -= 1

    return out
#------------ Calculamos la entropía y gradiente espacial de cada capa en el dataset --------------------
def entropia_gradiente(tupla_datos):
    inicio , matriz =tupla_datos
    #aquí guardamos resultados
    r =[]
    #contamos las capas z q nos mandaron
    capas = matriz.shape[1]

    for i in range(capas):
        #primero para entropía
        indice = inicio + i
        pixeles = matriz[:, i, :, :].flatten().astype(np.uint8)
        #frecuencias de intensidad (0 a 255)
        conteos = np.bincount(pixeles, minlength=256)
        #probabilidad (el número de casos donde aparece sobre el numero de casos totales)
        p = conteos / np.sum(conteos)
        # solo tomamos las probabilidades distintas de 0
        p = p[p>0]
        # Entropia: - sum(p * log2(p))
        entropia = -np.sum(p * np.log2(p))
        #ahora para el gradiente
        capa_f = matriz[:, i, :, :].astype(np.float32)
        # Diferencias absolutas en los ejes espaciales Y, X
        gy_abs = np.abs(np.diff(capa_f, axis=1))
        gx_abs = np.abs(np.diff(capa_f, axis=2))

        # Extraer el borde MÁS fuerte de cada imagen individualmente
        # axis=(1, 2) colapsa Y y X, dejando un arreglo de tamaño N
        picos_gy = gy_abs.max(axis=(1, 2))
        picos_gx = gx_abs.max(axis=(1, 2))

        # Promedio poblacional de los picos máximos
        gradiente = (picos_gy.mean() + picos_gx.mean()) / 2.0

        r.append((indice, entropia, gradiente))

    return r


def init_centroides(datos, cant_centroides):
    valores = [d[0] for d in datos]
    if cant_centroides == 2:
        return [[min(valores)], [max(valores)]]

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
        #usamos [c[:] for c in centroides] para asegurar una copia real de los valores
        centroides_anteriores = [c[:] for c in centroides]
        centroides = actualizar_centroides(datos, etiquetas, K, centroides_anteriores)
        cambio = movimiento_centroides(centroides_anteriores, centroides, K)
        if cambio < tol:
            break
    return centroides, etiquetas

if __name__ == '__main__':
    #cargamos el dataset de manera local
    archivo_local = np.load('fracturemnist3d.npz')
    n_p =3

    #juntamos las 3 particiciones del dataset oficial en una sola
    volumenes = np.concatenate((
        archivo_local['train_images'],
        archivo_local['val_images'],
        archivo_local['test_images']
    ), axis=0)

    #mismas etiquetas
    etiquetas = np.concatenate((
        archivo_local['train_labels'],
        archivo_local['val_labels'],
        archivo_local['test_labels']
    ), axis=0)

    N, Z, Y, X = volumenes.shape      #dimensiones de cada cosa

    #aquí se harán las pruebas y mediciones para el speed up y eficiencia

    cantidad_procesadores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    tiempos = {}  #donde va a ir el resultado de cada iteración
    resultados_optimos = None

    print(f"{'Procesadores'} \t | {'Tiempo (s)'} \t | {'Speed-up'} \t | {'Eficiencia'}")

    #iteramos sobre la cantidad de procesadores
    for p in cantidad_procesadores:
        tareas = Nivelar(p, Z, volumenes)

        inicio = time.time()
        with mp.Pool(processes=p) as pool:
            resultados = pool.map(entropia_gradiente, tareas)
        fin = time.time()
        t_ejecucion = fin - inicio
        tiempos[p] = t_ejecucion
        # El tiempo secuencial siempre es el del primer escenario (p=1)
        t1 = tiempos[1]

        speedup = t1 / t_ejecucion
        eficiencia = speedup / p

        print(f"{p} \t \t \t \t| {t_ejecucion:.3f} \t \t | {speedup:.4f} \t \t | {eficiencia:.4f}")

        #conservar los resultados para el K-Means
        resultados_optimos = resultados

    print("-------------------------------------------------------------------------")

    metricas_finales = []
    for lista_local in resultados_optimos:
        metricas_finales.extend(lista_local)

    metricas_finales.sort(key=lambda x: x[0])

    valores_entropia = [[m[1]] for m in metricas_finales]
    valores_gradiente = [[m[2]] for m in metricas_finales]   #separamos para hacer 2 kmeans por separado

    K = 2
    margen_error = 0.001

    centroides_finales_e, etiquetas_finales_e = ejecutar_kmeans_lineal(valores_entropia, K, margen_error)   #kmeans para entropía
    centroides_finales_e.sort(key=lambda c: c[0])

    centroides_finales_g, etiquetas_finales_g = ejecutar_kmeans_lineal(valores_gradiente, K, margen_error)  #kmeans para gradiente espacial
    centroides_finales_g.sort(key=lambda c: c[0])

    umbral_entropia = (centroides_finales_e[0][0] + centroides_finales_e[1][0]) / 2
    umbral_gradiente = (centroides_finales_g[0][0] + centroides_finales_g[1][0]) / 2

    print(f"Umbral calculado para entropía: {umbral_entropia:.4f}")
    print(f"Umbral calculado para gradiente: {umbral_gradiente:.4f}")
    print (f"Etiquetas por capa (entropía): {etiquetas_finales_e}")
    print(f"Etiqueta por capa (gradiente): {etiquetas_finales_g}")


    capas_utiles = 0
    indices_retenidos = []
    #decisión de capas útiles
    for indice, entropia, gradiente in metricas_finales:
        #consenso por unión
        supera_entropia = entropia > umbral_entropia
        supera_gradiente = gradiente > umbral_gradiente

        util = supera_entropia or supera_gradiente

        if util:
            capas_utiles += 1
            indices_retenidos.append(indice)

        print(f"Z={indice} \t {entropia:.3f} \t {gradiente:.3f} \t {'útil' if util else 'fondo'}")

    print(f"\nCapas útiles: {capas_utiles} de {Z}")

    z_min = min(indices_retenidos)
    z_max = max(indices_retenidos)

    volumenes_recortados = volumenes[:, z_min:z_max + 1, :, :]

    directorio_salida = "dataset_procesado"
    os.makedirs(directorio_salida, exist_ok=True)

    ruta_archivo = os.path.join(directorio_salida, "fracturemnist_recortado.npz")
    np.savez_compressed(ruta_archivo, images=volumenes_recortados, labels=etiquetas)

    print(f"\nDataset recortado guardado en: {ruta_archivo}")

