import numpy as np
import math
import os
from multiprocessing import Pool
from sklearn.model_selection import  StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report

#Entropía 
def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropia = 0
    for i in probabilities:
        entropia += i * (math.log(i, 2))
    return -entropia

#Ganancia
def information_gain(parent_labels, left_labels, right_labels):
    parent_entropy = entropy(parent_labels)
    left_weight = len(left_labels) / len(parent_labels)
    right_weight = len(right_labels) / len(parent_labels)
    hijo1_entropy = entropy(left_labels)
    hijo2_entropy = entropy(right_labels)
    gain = parent_entropy - (hijo1_entropy * left_weight + hijo2_entropy * right_weight)
    return gain

# --- CONSTRUCCION  DEL ÁRBOL ---
def split(X, y, feat_idxs):
    #establecemos los umbrales
    best_gain = -1
    split_idx, split_thresh = None, None

    #por cada vóxel
    for feat_idx in feat_idxs:
        #comparar el vóxel con  el mismo de las demás imágenes
        X_column = X[:, feat_idx]
        thresholds = np.unique(X_column)
        #buscamos el mejor umbral para ese vóxel
        for threshold in thresholds:
            left_idxs = X_column <= threshold
            right_idxs = X_column > threshold
            if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
                continue
            gain = information_gain(y, y[left_idxs], y[right_idxs])
            if gain > best_gain:
                best_gain = gain
                split_idx = feat_idx
                split_thresh = threshold
    return split_idx, split_thresh #devolvemos el índice del vóxel y el umbral que mejor separa las clases

#usamos recursión para construir el árbol completo
def construir_arbol(X, y, profundidad=0, max_profundidad=5, min_samples_split=2, max_features=None):
    #numero de imagenes, número de vóxeles
    ejemplos, voxeles = X.shape
    #clases
    n_labels = len(np.unique(y))
    #establecer el número de características a considerar en cada división
    if max_features is None:
        max_features = int(np.sqrt(voxeles))

    # CASO BASE: Si alcanzamos la profundidad máxima, si todas las muestras pertenecen a la misma clase o si no hay suficientes muestras para dividir, hacemos una hoja
    if profundidad >= max_profundidad or n_labels <= 1 or ejemplos < min_samples_split:
        valores, conteos = np.unique(y, return_counts=True)
        leaf_value = valores[np.argmax(conteos)]
        return {'leaf': True, 'value': leaf_value}
    
    # Seleccionar aleatoriamente un subconjunto de características para esta división
    caracteristicas  = np.random.choice(voxeles, max_features, replace=False)
    #vemos cuál es la mejor característica y umbral para dividir los datos
    mejor_car, umbral = split(X, y, caracteristicas )
    
    # Si no se encontró una división válida, hacemos una hoja
    if mejor_car is None:
        valores, conteos = np.unique(y, return_counts=True)
        leaf_value = valores[np.argmax(conteos)]
        return {'leaf': True, 'value': leaf_value}
    
    left_idxs = X[:, mejor_car] <= umbral
    right_idxs = X[:, mejor_car] > umbral
    
    left = construir_arbol(X[left_idxs, :], y[left_idxs], profundidad + 1, max_profundidad, min_samples_split, max_features)
    right = construir_arbol(X[right_idxs, :], y[right_idxs], profundidad + 1, max_profundidad, min_samples_split, max_features)
    
    return {'leaf': False, 'feature': mejor_car, 'threshold': umbral, 'left': left, 'right': right}

def predict_row(row, tree):
    if tree['leaf']:
        return tree['value']
    if row[tree['feature']] <= tree['threshold']:
        return predict_row(row, tree['left'])
    return predict_row(row, tree['right'])

def predict_tree(X, tree):
    return np.array([predict_row(row, tree) for row in X])


def predict_random_forest(X, forest):
    tree_preds = np.array([predict_tree(X, tree) for tree in forest])
    tree_preds = np.swapaxes(tree_preds, 0, 1)
    y_pred = [np.bincount(sample_preds).argmax() for sample_preds in tree_preds]
    return np.array(y_pred)

#Nivelamos cargas
def Nivelar(n_p, n_trees_totales):
    s = n_trees_totales % n_p
    t = n_trees_totales // n_p
    arboles_por_proceso = []
    
    for i in range(n_p):
        # Si sobran árboles, los repartimos 1 a 1 en los primeros procesos
        arboles_asignados = t + 1 if s > 0 else t
        if arboles_asignados > 0:
            arboles_por_proceso.append(arboles_asignados)
        s -= 1
    return arboles_por_proceso

#esta función se ejecutará en cada proceso para entrenar su parte del bosque
def entrenar_arboles_worker(num_arboles, X, y, max_depth, min_samples_split, max_features):
    np.random.seed(os.getpid()) 
    
    sub_forest = []
    n_samples = X.shape[0]
    
    for _ in range(num_arboles):
        # Bootstrap
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        X_sample, y_sample = X[idxs], y[idxs]
        
        # Construir árbol
        tree = construir_arbol(X_sample, y_sample, profundidad=0, 
                          max_profundidad=max_depth, 
                          min_samples_split=min_samples_split, 
                          max_features=max_features)
        sub_forest.append(tree)
        
    return sub_forest

if __name__ == "__main__":
    #carga de datos filtrados
    with np.load('dataset_procesado/fracturemnist_recortado.npz') as data:
        X_raw = data['images']
        y = data['labels'].flatten().astype(int)
    
    # Aplanar las imágenes a 2D para el RF
    n_muestras = X_raw.shape[0]
    X_filtrado = X_raw.reshape(n_muestras, -1)
     
    #datos sin filtrar
    with np.load('fracturemnist3d_completo.npz') as data:
        X_raw = data['images']
    
    #aplanar 
    n_muestras = X_raw.shape[0]
    X_completo = X_raw.reshape(n_muestras, -1)

    
    # Configuración del bosque (solo para pruebas)
    n_trees_totales = 150
    max_depth = 10
    min_samples_split = 2
    max_features = None
    
    #trabajamos con todos nucleos lógicos
    n_p = os.cpu_count() 
    K_FOLDS = 5
    
    print(f"\n--- {K_FOLDS}-FOLD CROSS VALIDATION ---")
    print(f"Árboles por bosque: {n_trees_totales} | Núcleos: {n_p}")
    
    cv_accuracies = []
    cv_f1_scores = []

    cv_accuracies_com = []
    cv_f1_scores_com = []

    # shuffle=True es muy importante para mezclar las imágenes antes de dividir
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    all_y_true = []
    all_y_pred = []
    all_y_true_com = []
    all_y_pred_com = []
    
    
    # skf.split() devuelve automáticamente los índices de entrenamiento y prueba
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_filtrado, y), 1):
        print(f"\n--- Procesando Fold {fold_idx}/{K_FOLDS} ---")
        
        # filtrado
        X_train, y_train = X_filtrado[train_idx], y[train_idx]
        X_test, y_test = X_filtrado[test_idx], y[test_idx]

        X_train_com, y_train_com = X_completo[train_idx], y[train_idx]#completo
        X_test_com, y_test_com = X_completo[test_idx], y[test_idx]
        
        # configuración
        arboles_distribuidos = Nivelar(n_p, n_trees_totales)
        tareas = [(num, X_train, y_train, max_depth, min_samples_split, max_features) for num in arboles_distribuidos]
        #completo
        tareas_com = [(num, X_train_com, y_train_com, max_depth, min_samples_split, max_features) for num in arboles_distribuidos]

        # 3. Entrenar en Paralelo
        with Pool(n_p) as p:
            resultados = p.starmap(entrenar_arboles_worker, tareas)
            resultados_com = p.starmap(entrenar_arboles_worker, tareas_com)
            
        forest = [tree for sub_forest in resultados for tree in sub_forest]
        forest_com = [tree for sub_forest in resultados_com for tree in sub_forest]

        # 4. Predecir y Evaluar
        y_pred = predict_random_forest(X_test, forest)
        y_pred_com = predict_random_forest(X_test_com, forest_com)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_true_com.extend(y_test_com)
        all_y_pred_com.extend(y_pred_com)

        fold_acc = np.mean(y_pred == y_test)
        fold_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        fold_acc_com = np.mean(y_pred_com == y_test_com)
        fold_f1_com = f1_score(y_test_com, y_pred_com, average='weighted', zero_division=0) 
        
        print(f"Fold {fold_idx} - Accuracy: {fold_acc:.4f} | F1-Score: {fold_f1:.4f}")
        print(f"Fold {fold_idx} (Completo) - Accuracy: {fold_acc_com:.4f} | F1-Score: {fold_f1_com:.4f}")
        
        cv_accuracies.append(fold_acc)
        cv_f1_scores.append(fold_f1)

        cv_accuracies_com.append(fold_acc_com)
        cv_f1_scores_com.append(fold_f1_com)

# --- RESULTADOS FINALES DETALLADOS ---


    print("\n      MÉTRICAS DEL MODELO FILTRADO")
    #Matriz de confusión
    conf_matrix = confusion_matrix(all_y_true, all_y_pred)
    print("\nMatriz de Confusión:")
    print(conf_matrix)
    
    # Precision, Recall, F1 por clase
    print("\nMétricas por Clase:")
    print(classification_report(all_y_true, all_y_pred))
    

    print("\n=============================================")
    print(f"Accuracy Promedio Final: {np.mean(cv_accuracies):.4f}")
    print(f"F1-Score Promedio Final: {np.mean(cv_f1_scores):.4f}")


    conf_matrix_com = confusion_matrix(all_y_true_com, all_y_pred_com)
    print("\nMatriz de Confusión (Completo):")
    print(conf_matrix_com)

    print("\nMétricas por Clase (Completo):")
    print(classification_report(all_y_true_com, all_y_pred_com))

    print("\n=============================================")
    print(f"Accuracy Promedio Final (Completo): {np.mean(cv_accuracies_com):.4f}")
    print(f"F1-Score Promedio Final (Completo): {np.mean(cv_f1_scores_com):.4f}")
    print("=============================================")  

    