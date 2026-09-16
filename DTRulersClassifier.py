import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from imblearn.metrics import geometric_mean_score
#from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import numpy as np
from sklearn.metrics import accuracy_score   # o la que uses
import seaborn as sns
from xgboost import plot_tree
import matplotlib.pyplot as plt

class DTRulersClassifier:

    def __init__(self, random_state, max_depth):
        """
        Constructor de la clase DTRulers.

        :param random_state: Semilla para reproducibilidad.
        :param max_depth: Profundidad máxima para XGBoost y DecisionTreeClassifier.
        """
        self.random_state = random_state
        self.max_depth = max_depth

        self.rules = []  # Reglas extraídas del modelo XGBoost
        # self.decision_tree = DecisionTreeClassifier(random_state=self.random_state, max_depth=self.max_depth)
        self.decision_tree = DecisionTreeClassifier(random_state=self.random_state, max_depth=self.max_depth)  
        self.transformed_X = None  # Dataset modificado durante el entrenamiento
        self.xgb_model_f = None # Modelo XGBoost entrenado
        self.feature_importances = {}  # Inicializar el atributo feature_importances como un diccionario vacío
        self.rules_per_feature = {}  # Nuevo atributo para almacenar el conteo de reglas
        self.rng = np.random.RandomState(random_state)  # Crear el objeto RandomState
        self.transformed_X_AUX = None # Lista de reglas únicas extraídas del modelo XGBoost
        self.rules_AUX =[]
        self.unique_rules_set = []  # Lista de reglas únicas extraídas del modelo XGBoost

    def fit(self, X, y):
        """
        Entrena el clasificador propio basado en reglas.

        :param X: DataFrame de características.
        :param y: Serie o array de la variable objetivo.## ojoo
        
        # Eliminar características correlacionadas
        # X = self._remove_correlated_features(X, threshold=0.8)
        # Entrenar el modelo XGBoost
        """
        xgb_model = xgb.XGBClassifier(max_depth=self.max_depth, random_state=self.random_state)
        xgb_model.fit(X, y)

        # Extraer reglas del modelo XGBoost auxiliar
        self.rules_AUX, feature_importances = self._extract_rules_from_xgboost_aux(xgb_model,X,y)
        
        # Modificar el dataset original basado en las reglas extraídas
        self.transformed_X_AUX = self._apply_rules_to_dataset(X, self.rules_AUX)
        
        self.xgb_model_f = xgb.XGBClassifier(max_depth=self.max_depth, random_state=self.random_state)
        self.xgb_model_f.fit(self.transformed_X_AUX, y)
        
        self.rules, self.feature_importances = self._extract_rules_from_xgboost(xgb_model=self.xgb_model_f, X_test=self.transformed_X_AUX, y_test=y)
        
        # Verificar si las reglas finales están vacías
        if not self.rules:
            #print("Advertencia: Las reglas finales (rules) están vacías.")
            self.rules = self.rules_AUX[:]
        self.transformed_X = self._apply_rules_to_dataset_aux(self.transformed_X_AUX, self.rules)# CAMBIAR transformed_X POR X
       
        return self.decision_tree.fit(self.transformed_X, y)

    def predict(self, X):
        """
        Realiza predicciones usando el clasificador propio.

        :param X: DataFrame de características para predicción.
        :return: Predicciones como array.
        """

        # Transformar el conjunto de predicción utilizando las mismas reglas
        transformed_X_AUX = self._apply_rules_to_dataset(X, self.rules_AUX)
        # # print(" \n transformed_X_AUX desde predict desde PREDIT : ", transformed_X_AUX.columns)
        transformed_X = self._apply_rules_to_dataset_aux(transformed_X_AUX, self.rules)

        return self.decision_tree.predict(transformed_X)

    def _extract_rules_from_xgboost_aux(self, xgb_model, X_test, y_test):
        """
        Extrae las reglas de un modelo XGBoost, seleccionando un porcentaje basado en la
        importancia por permutación.

        Args:
            xgb_model: Modelo XGBoost entrenado.
            X: DataFrame de características.
            y: Serie de la variable objetivo.
            target_percentage: Porcentaje de reglas a seleccionar (por defecto 0.1).

        Returns:
            Lista de reglas únicas y ordenadas, diccionario de conteo de reglas por característica.
        """
        tree_dumps = xgb_model.get_booster().get_dump()
        self.unique_rules_set = []
        # Extraer reglas únicas
        for tree_dump in tree_dumps:
            for line in tree_dump.split('\n'):
                if 'leaf=' in line:
                    continue  # Ignorar las hojas
                if '[' in line:
                    rule = line.split('[')[1].split(']')[0]
                    if rule not in self.unique_rules_set:  # Evitar duplicados manteniendo el orden
                        self.unique_rules_set.append(rule)
       
        filtered_rules=self.unique_rules_set
        importances_promedio =[]
        return filtered_rules, importances_promedio

    def _extract_rules_from_xgboost(self, xgb_model, X_test, y_test, target_percentage= 1):
        """
        Extrae las reglas de un modelo XGBoost, seleccionando un porcentaje basado en la
        importancia por permutación.

        Args:
            xgb_model: Modelo XGBoost entrenado.
            X: DataFrame de características.
            y: Serie de la variable objetivo.
            target_percentage: Porcentaje de reglas a seleccionar (por defecto 0.1).

        Returns:
            Lista de reglas únicas y ordenadas, diccionario de conteo de reglas por característica.
        """
        tree_dumps = xgb_model.get_booster().get_dump()
        unique_rules_set = []
        # Extraer reglas únicas
        for tree_dump in tree_dumps:
            for line in tree_dump.split('\n'):
                if 'leaf=' in line:
                    continue  # Ignorar las hojas
                if '[' in line:
                    rule = line.split('[')[1].split(']')[0]
                    if rule not in unique_rules_set:  # Evitar duplicados manteniendo el orden
                        unique_rules_set.append(rule)

        rules_per_feature = {feature: 0 for feature in X_test.columns}

        #--- Calcular importancia por permutación VARIAS FORMAS --------------#
        def seleccionar_metrica(y):
            """
            Selecciona la métrica adecuada según las características del problema.

            Args:
                y: Serie o array de la variable objetivo.

            Returns:
                Una función de métrica adecuada.
            """
            # Verificar si el problema es balanceado
            class_counts = y.value_counts(normalize=True)
            imbalance_ratio = class_counts.max() / class_counts.min()

            if imbalance_ratio < 1.5:  # Problema balanceado
                #print("Problema balanceado detectado. Usando 'accuracy_score'.")
                return accuracy_score
            elif imbalance_ratio >= 1.5 and imbalance_ratio < 5:  # Moderadamente desbalanceado
                #print("Problema moderadamente desbalanceado detectado. Usando 'balanced_accuracy_score'.")
                return balanced_accuracy_score
            elif imbalance_ratio >= 5:  # Altamente desbalanceado
                #print("Problema altamente desbalanceado detectado. Usando 'geometric_mean_score'.")
                return geometric_mean_score
            else:
                #print("Usando métrica por defecto: 'accuracy_score'.")
                return accuracy_score
       
        # Seleccionar la métrica adecuada según el dataset
        metric = seleccionar_metrica(y_test)
        
        importances_promedio  = self.permutation_importance(
            xgb_model,
            X_test,
            y_test,
            metric=metric,
            n_repeats=5,
            threshold=0.8,
            random_state= self.random_state,
            sample_frac=1,
        )
       
        # --- Filtrar características con importancia mayor a 0 ---
        importances_filtradas = {f: v for f, v in importances_promedio.items() if v > 0.0}
        # --- Normalizar importancias ---
        total_importance = sum(importances_filtradas.values())  # Sumar solo las importancias > 0
        #print("total_importance ", total_importance)
        normalized_importances = {feature: imp / total_importance for feature, imp in importances_filtradas.items()}
        #print("normalized_importances ", normalized_importances)
        # --- Seleccionar características por importancia acumulada ---
        cumulative_importance = 0.0
        selected_features = []
        # Ordenar características por importancia normalizada (de mayor a menor)
        sorted_features = sorted(normalized_importances, key=normalized_importances.get, reverse=True)
        for feature in sorted_features:
            feature_importance = normalized_importances[feature]
            cumulative_importance += feature_importance  # Acumulando la importancia
            #print("cumulative_importance__ ", cumulative_importance)
            selected_features.append(feature)
            if cumulative_importance > target_percentage:  # Se detiene al llegar al límite
                break
        
        # --- Filtrar reglas cuya importancia correspondiente sea > 0 y estén seleccionadas ---
        selected_features_set = set(selected_features) # Nombres como 'sepal length (cm)_lt_5.5999999' con importancia > 0
        filtered_rules = []
        for rule in unique_rules_set: # Reglas como 'sepal length (cm)_lt_5.5999999<1'

                # La característica que usa esta regla es la parte ANTES del último '<'
                # Tenemos que encontrar el último '<' para separar correctamente
                last_lt_index = rule.rfind('<') # rfind encuentra la última ocurrencia
                if last_lt_index != -1: # Asegurarse de que se encontró '<'
                    # La característica usada por la regla es todo lo anterior al último '<'
                    feature_used_by_rule = rule[:last_lt_index].strip()

                    # Ahora verificamos si esta característica está en nuestro conjunto seleccionado
                    if feature_used_by_rule in selected_features_set:
                        filtered_rules.append(rule) # Añadimos la regla completa original

                # print(" \n filtered_rules desde _extract_rules_from_xgboost \n: ", filtered_rules)
                # print(len(filtered_rules))
        # Contar las reglas por característica (usando filtered_rules)
        for rule in filtered_rules:
            for feature in selected_features:
                if feature in rule:
                    rules_per_feature[feature] += 1
        # Almacenar el conteo de reglas por característica en el atributo de la clase
        self.rules_per_feature = rules_per_feature
        return filtered_rules, importances_promedio

    def _apply_rules_to_dataset(self, dataset, rules):
        """
        Transforma un dataset aplicando las reglas de manera vectorizada.
        """
        new_columns = {}

        # Crear un DataFrame vacío para las nuevas columnas
        for rule in rules:
            feature_name, threshold = rule.split('<')
            feature_name = feature_name.strip()
            threshold = float(threshold.strip())

            if feature_name in dataset.columns:
                new_column_name = f"{feature_name}_lt_{threshold}"
                new_columns[new_column_name] = (dataset[feature_name] < threshold).astype(int)

        # Crear un nuevo DataFrame con las columnas generadas
        return pd.concat([dataset, pd.DataFrame(new_columns, index=dataset.index)], axis=1)
        #return pd.DataFrame(new_columns)

    def _apply_rules_to_dataset_aux(self, dataset, rules):
        """
        Transforma un dataset aplicando las reglas de manera vectorizada.
        """
        new_columns = {}

        # Crear un DataFrame vacío para las nuevas columnas
        for rule in rules:
            feature_name, threshold = rule.split('<')
            feature_name = feature_name.strip()
            threshold = float(threshold.strip())

            if feature_name in dataset.columns:
                new_column_name = f"{feature_name}_lt_{threshold}"
                new_columns[new_column_name] = (dataset[feature_name] < threshold).astype(int)

        # Crear un nuevo DataFrame con las columnas generadas
        #return pd.concat([dataset, pd.DataFrame(new_columns, index=dataset.index)], axis=1)
        return pd.DataFrame(new_columns)

    def visualize_tree_text(self):
        """
        Visualiza el árbol de decisión en forma de texto.
        """
        tree_text = export_text(self.decision_tree, feature_names=list(self.transformed_X.columns))
        print("Árbol de Decisión en Texto:")
        print(tree_text)

    def visualize_tree_plot(self):
        """
        Visualiza el árbol de decisión en forma de dibujo.
        """
        plt.figure(figsize=(20,10))
        class_names = [str(cls) for cls in self.decision_tree.classes_]
        plot_tree(self.decision_tree, feature_names=list(self.transformed_X.columns), class_names=class_names,filled=True,
            rounded=True,
            fontsize=9 )
        plt.show()



    def _predict_labels(self,model, X_np):
        """
        Retorna etiquetas de clase (int) para cualquier clasificador:
        - sklearn, LightGBM, XGBoost scikit-wrapper
        - Booster nativo de XGBoost
        """
        if hasattr(model, "predict"):            # sklearn / XGBClassifier
            pred = model.predict(X_np)
            # En XGBClassifier multiclase, pred.shape -> (n, n_classes)
            if pred.ndim == 2:
                pred = pred.argmax(1)
        else:                                    # Booster puro
            pred = model.inplace_predict(X_np, predict_type="class")
        return np.asarray(pred, dtype=np.int64)

    def permutation_importance(self, model, X_test, y_test,
                                metric, n_repeats, threshold,
                                random_state, sample_frac):
        # 1) muestreo ----------------------------------------------------------------
        Xs = X_test.sample(frac=sample_frac, random_state=random_state)
        ys = y_test.loc[Xs.index].to_numpy()
        X_np = np.ascontiguousarray(Xs.to_numpy(dtype=np.float32))
        

        # 2) baseline ----------------------------------------------------------------
        baseline_pred = self._predict_labels(model, X_np)
        baseline = metric(ys, baseline_pred)

        # 3) correlaciones -----------------------------------------------------------
        corr = np.abs(np.corrcoef(X_np, rowvar=False))

# ...existing code...
        corr = np.abs(np.corrcoef(X_np, rowvar=False))
        # Visualización de la matriz de correlación
        plt.figure(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", xticklabels=Xs.columns, yticklabels=Xs.columns)
        plt.title("Matriz de correlación (absoluta)")
        plt.tight_layout()
        plt.show()
# ...existing code...



        corr_idx = {i: np.where(corr[i] > threshold)[0] for i in range(corr.shape[0])}
        rng   = np.random.default_rng(random_state)
        imps  = np.zeros(X_np.shape[1], dtype=np.float64)

        # 4) permutaciones -----------------------------------------------------------
        for _ in range(n_repeats):
            for i in range(X_np.shape[1]):
                backup_i   = X_np[:, i].copy()
                backup_corr = {j: X_np[:, j].copy() for j in corr_idx[i] if j != i}

                for j in corr_idx[i]:
                    if j != i:
                        X_np[:, j] = 0.0

                rng.shuffle(X_np[:, i])


                perm_pred = self._predict_labels(model, X_np)
                imps[i]  += baseline - metric(ys, perm_pred)
                # restaurar
                X_np[:, i] = backup_i
                for j, v in backup_corr.items():
                    X_np[:, j] = v

        imps /= n_repeats
        return dict(sorted(zip(Xs.columns, imps), key=lambda kv: -kv[1]))


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evalúa un modelo en términos de Accuracy, Balanced Accuracy, G-mean, Precision, Recall, y F1-score.
    Suprime advertencias de división por cero asignando 0 en esos casos.

    Args:
        model: Modelo entrenado.
        X_train, X_test: Datos de entrenamiento y prueba.
        y_train, y_test: Etiquetas de entrenamiento y prueba.

    Returns:
        Un diccionario con las métricas calculadas.
    """

    # Predicciones en conjunto de prueba
    y_pred = {
        "train": model.predict(X_train),
        "test": model.predict(X_test)
    }

    # Detectar si es un dataset binario o multiclase
    num_classes = len(np.unique(y_train))
    average_type = 'binary' if num_classes == 2 else 'macro'

    # Definir las métricas a calcular
    metric_functions = {
        "Accuracy": accuracy_score,
        "Balanced_Accuracy": balanced_accuracy_score,
        "G_mean": geometric_mean_score,
        "Precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average=average_type, zero_division=0),
        "Recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average=average_type, zero_division=0),
        "F1_Score": lambda y_true, y_pred: f1_score(y_true, y_pred, average=average_type, zero_division=0)
    }

    # Calcular las métricas para entrenamiento y prueba
    metrics = {}
    for metric_name, metric_func in metric_functions.items():
        metrics[f"{metric_name}_Train"] = metric_func(y_train, y_pred["train"])
        metrics[f"{metric_name}_Test"] = metric_func(y_test, y_pred["test"])

    return metrics

####################################### PARALELIZAR OPTIMIZADO  ########################################
# # import os, itertools, pandas as pd
# # from joblib import Parallel, delayed
# # from sklearn.model_selection import train_test_split
# # from sklearn.tree import DecisionTreeClassifier
# # import xgboost as xgb

# # --------- 1. limitar hilos internos ----------
# # os.environ["OMP_NUM_THREADS"]     = "1"
# # os.environ["OPENBLAS_NUM_THREADS"] = "1"
# # os.environ["MKL_NUM_THREADS"]      = "1"
# # os.environ["NUMEXPR_NUM_THREADS"]  = "1"
# # ----------------------------------------------

# # def run_experiment(dataset_path, dataset_name,
# #                    random_state, max_depth, cls_name):

# #     leemos una sola vez (mem-map para I/O grande)
# #     data = pd.read_csv(dataset_path, memory_map=True)
# #     y = data["class"]
# #     X = data.drop(columns=["class"])

# #     Xtr, Xts, ytr, yts = train_test_split(
# #         X, y, test_size=0.2, random_state=random_state, stratify=y)

# #     --- instanciar modelo -------------
# #     if cls_name == "DTRulersClassifier":
# #         model = DTRulersClassifier(random_state=random_state,
# #                                    max_depth=max_depth,
# #                                    )
# #     elif cls_name == "XGBoost":
# #         model = xgb.XGBClassifier(max_depth=max_depth,
# #                                   n_jobs=1,
# #                                   random_state=random_state)
# #     elif cls_name == "DecisionTreeClassifier":
# #         model = DecisionTreeClassifier(max_depth=max_depth,
# #                                        random_state=random_state)

# #     model.fit(Xtr, ytr)
# #     metrics = evaluate_model(model, Xtr, Xts, ytr, yts)

# #     extras sólo para DTRulers
# #     depth = model.decision_tree.get_depth() if cls_name=="DTRulersClassifier" else None
# #     rules_i = len(getattr(model, "rules_AUX", []))
# #     rules_f = len(getattr(model, "rules", []))

# #     return {
# #         "Dataset"      : dataset_name,
# #         "Classifier"   : cls_name,
# #         "Random State" : random_state,
# #         "Max Depth"    : max_depth,
# #         "Tree Depth"   : depth,
# #         "Rules_ini"    : rules_i,
# #         "Rules_fin"    : rules_f,
# #         **metrics
# #     }

# # ---------------- pool de trabajos ----------------
# # datasets_folder = "10_datasets"
# # dataset_files   = [f for f in os.listdir(datasets_folder) if f.endswith(".csv")]
# # classifiers     = ["DTRulersClassifier"]         # añade otros si quieres
# # random_states   = range(1, 32)
# # max_depths      = range(3, 9)

# # jobs = []
# # for f in dataset_files:
# #     path = os.path.join(datasets_folder, f)
# #     name = f.split(".")[0]
# #     for rs, md, cls in itertools.product(random_states,
# #                                          max_depths,
# #                                          classifiers):
# #         jobs.append((path, name, rs, md, cls))

# # print(f"Lanzando {len(jobs):,d} tareas en paralelo …")

# # results = Parallel(n_jobs=-1, backend="loky",
# #                    verbose=10, batch_size="auto")(
# #             delayed(run_experiment)(*job) for job in jobs)

# # pd.DataFrame(results).to_csv("results_fast_solo_reglas.csv", index=False)

################################## CASO PARTICULAR #############################################################################

if __name__ == '__main__':
    # tiempo_total =  time.time()
    # # Medir el tiempo para leer el archivo CSV
    # start_time = time.time()
    file_path = "10_datasets\iris_complete.csv"  # Cambia esto por la ruta a tu archivo CSV
    data = pd.read_csv(file_path)
    print(data.head()) # Mostrar las primeras filas del DataFrame
    # #print(f"Tiempo para leer el archivo CSV: {time.time() - start_time:.2f} segundos")

    # # Medir el tiempo para dividir las características y la columna objetivo
    # start_time = time.time()
    target_column = "class"
    #eliminar_petal_length="petal length (cm)"
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # #print(f"Tiempo para dividir características y objetivo: {time.time() - start_time:.2f} segundos")

    # # Normalizar las características
    # # scaler = StandardScaler()
    # # X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    # # print(X_normalized)

    # # Ahora X_normalized contiene los datos normalizados
    # # Medir el tiempo para dividir en entrenamiento y prueba
    # start_time = time.time()
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)
    #print(f"Tiempo para dividir en entrenamiento y prueba: {time.time() - start_time:.2f} segundos")
    # for col in ['seismic', 'seismoacoustic', 'shift', 'ghazard']:
    #     X_train[col] = pd.Categorical(X_train[col])
    #     X_test[col] = pd.Categorical(X_test[col])
    # #Medir el tiempo para entrenar el modelo
    # start_time = time.time()
    a = DTRulersClassifier(random_state=7, max_depth=4)
    a.fit(X_train, y_train)
    # print(f"Tiempo para entrenar el modelo: {time.time() - start_time:.2f} segundos")

    # Medir el tiempo para evaluar el modelo
    # start_time = time.time()
    mt = evaluate_model(a, X_train, X_test, y_train, y_test)  # se agrega self
    # print(f"Tiempo para evaluar el modelo: {time.time() - start_time:.2f} segundos")

    # Imprimir métricas y detalles del modelo
    print(mt)

    print(a.decision_tree)
    print(f"Profundidad máxima del árbol de decisión: {a.decision_tree.get_depth()}")
    print(len(a.rules_AUX))
    print(len(a.rules))

    # print(f"Tiempo total de ejecución:{time.time() - tiempo_total:.2f} segundos")

    # a.visualize_tree_plot()
