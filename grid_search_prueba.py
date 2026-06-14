import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

#Toooooodas nuestras librerías
import numpy as np
import multiprocessing as mp
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
from imblearn.ensemble import BalancedRandomForestClassifier #intentamos también balancear


########################################################################################################################
def repartir(cant_datos, cant_procesadores):
    residuo = cant_datos % cant_procesadores
    num_divisiones = int(cant_datos / cant_procesadores)
    salida = []
    inicio = 0
    for i in range(cant_procesadores):
        tamanio = num_divisiones + (1 if i < residuo else 0)
        fin = inicio + (tamanio - 1)
        salida.append([inicio, fin])
        inicio = fin + 1
    return salida


def evaluar_modelo_parcial(rango, combo, tipo_modelo, X_data, y_train, X_test, y_test, is_shm=True, shape=None, dtype=None):
    #usamos memoria compartida para evitar copiar el train en cada proceso, lo que sería muy costoso en tiempo y memoria
    #Cada proceso accede a la misma región de memoria donde se encuentra el train, así ahorramos recursos
    if is_shm:
        #si is_shm es True, solo accedemos a la memoria compartida, no copiamos
        #exist_shm es el objeto de memoria compartida que nos permite acceder a la matriz de train
        exist_shm = shared_memory.SharedMemory(name=X_data)
        #el puntero a la matriz de train 
        X_train_ptr = np.ndarray(shape, dtype=dtype, buffer=exist_shm.buf)
    else:
        # en secuencial no usamos memoria compartida, así que simplemente asignamos el train a una variable local
        X_train_ptr = X_data
    
    resultados_locales = []
    inicio, fin = rango
    
    for i in range(inicio, fin + 1):
        params = combo[i]
        if tipo_modelo == "rf":#esgún el tipo de modelo en texto elegimos
            #incializamos el modelo según los parametros de la combinación  y con n_jobs=1 para evitar que cada modelo use paralelismo interno
            model = RandomForestClassifier(**params, random_state=42, n_jobs=1)
        elif tipo_modelo == "brf": #balanceado
            model = BalancedRandomForestClassifier(**params, random_state=42, n_jobs=1)
        elif tipo_modelo == "knn":
            model = KNeighborsClassifier(**params, n_jobs=1)
        elif tipo_modelo == "svm":
            model = SVC(**params)

        #entrenamos 
        model.fit(X_train_ptr, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="binary")
        resultados_locales.append((params, acc, f1))
    
    if is_shm:
        #"cerramos" la memoria compartida
        exist_shm.close()

    return resultados_locales

########################################################################################################################
def grid_search(modelo, param_prueba, X_train, y_train, X_test, y_test, n_jobs):
    #creamos todas las combinaciones de hiperparámetros a probar
    lista_param = list(param_prueba.keys())
    combinaciones = [dict(zip(lista_param, vals)) for vals in product(*param_prueba.values())]
    lista_tareas = repartir(len(combinaciones), n_jobs)

    # copiamos el train una sola vez  a la memoria  compartida y pasamos cada proceso para que puedan acceder sin copiarlo
    #reservamos memoria
    shm = shared_memory.SharedMemory(create=True, size=X_train.nbytes)
    #creamos la copia donde se va a guardar
    X_shared = np.ndarray(X_train.shape, dtype=X_train.dtype, buffer=shm.buf)
    #copiamos el train a la memoria compartida
    np.copyto(X_shared, X_train)

    #empaquetamos los argumentos para cada proceso y la memoria compartida
    argumentos = [(tarea, combinaciones, modelo, shm.name, y_train, X_test, y_test, True, X_train.shape, X_train.dtype)
        for tarea in lista_tareas]
    
    #Pool
    with mp.Pool(processes=n_jobs) as pool:
        resultados_anidados = pool.starmap(evaluar_modelo_parcial, argumentos)

    #cerramos memoria
    shm.close()
    #liberamos 
    shm.unlink()

    lista_final = [item for sublist in resultados_anidados for item in sublist]
    lista_final.sort(key=lambda r: (r[2], r[1]), reverse=True)
    return lista_final

########################################################################################################################
def grid_search_secuencial(modelo, param_prueba, X_train, y_train, X_val, y_val):
    lista_param = list(param_prueba.keys())
    combinaciones = [dict(zip(lista_param, vals)) for vals in product(*param_prueba.values())]

    resultados = []
    rango_completo = [0, len(combinaciones) - 1]
    #ejecutamos en el hilo principa
    evaluar_modelo_parcial(rango_completo, combinaciones, modelo, X_train, y_train, X_val, y_val, resultados)

    resultados.sort(key=lambda r: (r[2], r[1]), reverse=True)
    return resultados


########################################################################################################################
if __name__ == "__main__":

    # Hiperparámetros optimizados para alta dimensionalidad / PCA
    param_pruebas_rf = {"max_depth": [10, 20, None], "n_estimators": [50, 100, 200], "min_samples_split": [2, 5],
                        "min_samples_leaf": [1, 2], "max_features": ["sqrt", "log2"], "class_weight": ["balanced", "balanced_subsample", None]   #limitamos la cantidad de características
                        }
    param_pruebas_brf = {"max_depth": [5, 10, 20, None], "n_estimators": [50, 100, 200], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2],
        "sampling_strategy": ["all", "majority", "not minority"] #Estrategias de balanceo
    }

    param_pruebas_knn = {"n_neighbors": [3, 5, 7, 11], "metric": ["euclidean", "cityblock", "chebyshev"],
                        "weights": ["uniform", "distance"]}

    param_pruebas_svm = {
                        "kernel": ["linear", "rbf", "poly"], "C": [0.1, 1, 10, 100],
                        "gamma": ["scale", "auto", 0.01]}

    #a partir de aquí es donde empieza a cambiar #######################################################################

    ruta_dataset = os.path.join("dataset_procesado", "fracturemnist_recortado.npz")  #cargamos el dataset
    archivo_local = np.load(ruta_dataset)

    #hacemos la extracción y el aplanado de las 3 particiones del dataset
    X_train_raw = archivo_local['train_images'].reshape(archivo_local['train_images'].shape[0], -1)
    X_val_raw = archivo_local['val_images'].reshape(archivo_local['val_images'].shape[0], -1)
    X_test_raw = archivo_local['test_images'].reshape(archivo_local['test_images'].shape[0], -1)
    
    y_train = archivo_local['train_labels'].flatten()   #obtenemos las etiquetas
    y_val = archivo_local['val_labels'].flatten()
    y_test = archivo_local['test_labels'].flatten()

    #print(X_train_raw.shape)

    #Estandarizamos con un standar scaler (queremos hacer un PCA pq puede que el harsware que tenemos no sea suficiente)
    scaler = StandardScaler()
    X_train_escalado = scaler.fit_transform(X_train_raw)
    X_val_escalado = scaler.transform(X_val_raw)
    X_test_escalado = scaler.transform(X_test_raw)

    #PCA
    pca = PCA(n_components=0.95, random_state=42)
    X_train = pca.fit_transform(X_train_escalado)
    X_val = pca.transform(X_val_escalado)
    X_test = pca.transform(X_test_escalado)

    #definimos nuestros modelos
    modelos = [
        ("Random Forest", "rf", param_pruebas_rf),
        ("Balanced Random Forest", "brf", param_pruebas_brf), # <--- NUEVO EN LA LISTA
        ("KNN", "knn", param_pruebas_knn),
        ("SVM", "svm", param_pruebas_svm),
    ]

    #Evaluación mejorada pq la anterior estaba mal hecha
    for nombre, tipo, param_prueba in modelos:
        print(f"===================================================================")
        print(f"----------------> Evaluando modelo: {nombre} <---------------------")

        #mdición secuencial
        inicio_sec = time.perf_counter()
        res_secuencial = grid_search_secuencial(tipo, param_prueba, X_train, y_train, X_val, y_val)
        fin_sec = time.perf_counter()
        tiempo_secuencial = fin_sec - inicio_sec
        print(f"Tiempo secuencial: {tiempo_secuencial:.4f} s")

        #medición paralela
        for num_p in range(2, mp.cpu_count()):
            inicio_paralelo = time.perf_counter()
            res_paralelo = grid_search(tipo, param_prueba, X_train, y_train, X_val, y_val, num_p)
            fin_paralelo = time.perf_counter()
            tiempo_paralelo = fin_paralelo - inicio_paralelo

            speed_up = tiempo_secuencial / tiempo_paralelo
            eficiencia = speed_up / num_p

            print(f"\n--- Prueba con {num_p} procesadores ---")
            print(f"Tiempo paralelo:   {tiempo_paralelo:.4f} s")
            print(f"Speed-up:          {speed_up:.4f}x")
            print(f"Eficiencia:        {eficiencia:.4f} ({(eficiencia * 100):.2f}%)")

        mejor_resultado = res_paralelo[0]
        mejores_params, mejor_acc, mejor_f1, mejor_matriz, mejor_reporte = mejor_resultado
        print(f"\n---------> Mejores resultados para {nombre} <----------")
        print(f"Accuracy: {mejor_acc:.4f} | F1-Score: {mejor_f1:.4f}")
        print(f"Hiperparámetros: {mejores_params}")
        print("\nMatriz de confusión:")
        print(mejor_matriz)
        print("\nReporte de clasificación:")
        print(mejor_reporte)
