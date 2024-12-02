import os
import pandas as pd
from ProcesadorAudios import ProcesadorAudios
from Parametrizador import Parametrizador
from Knn import Knn
from sklearn.model_selection import train_test_split


class Puntuador:
    def __init__(self, k_min=3, k_max=50, n_splits=5):
        """
        Clase para evaluar el desempeño del modelo con validación cruzada.
        :param k_min: Valor mínimo de k en k-NN.
        :param k_max: Valor máximo de k en k-NN.
        :param n_splits: Número de particiones para validación cruzada.
        """
        # Rutas
        self.crudos_path = r"C:\Users\berni\Desktop\ProyectoIAaudiosRobusto\DB\Crudos"
        self.candidato_csv = r"C:\Users\berni\Desktop\ProyectoIAaudiosRobusto\Parametros\candidato_parametros.csv"
        self.parametros_db_path = r"C:\Users\berni\Desktop\ProyectoIAaudiosRobusto\Parametros\base_datos_aumentada_parametros.csv"
        self.procesador = ProcesadorAudios()
        self.parametrizador = Parametrizador()
        self.k_min = k_min
        self.k_max = k_max
        self.n_splits = n_splits
        self.resultados_k = {}  # Guardar puntajes para cada valor de k

    def cargar_base_datos(self):
        """
        Carga la base de datos aumentada.
        """
        if not os.path.exists(self.parametros_db_path):
            raise FileNotFoundError(f"La base de datos en {self.parametros_db_path} no existe. Procesa primero los audios.")
        return pd.read_csv(self.parametros_db_path)

    def obtener_etiqueta_desde_nombre(self, file_name):
        """
        Asigna una etiqueta según el nombre del archivo.
        """
        file_name_lower = file_name.lower()
        if "zanahoria" in file_name_lower:
            return "zanahoria"
        elif "papa" in file_name_lower:
            return "papa"
        elif "camote" in file_name_lower:
            return "camote"
        elif "berenjena" in file_name_lower:
            return "berenjena"
        else:
            return "desconocido"

    def evaluar_con_validacion_cruzada(self, k):
        """
        Evalúa la base de datos completa usando validación cruzada para un valor específico de k.
        :param k: Valor de k para k-NN.
        :return: Puntaje promedio para el valor de k.
        """
        print(f"\nEvaluando con validación cruzada para k = {k}...")
        knn = Knn(candidato_csv=self.candidato_csv, k=k)

        # Cargar la base de datos aumentada
        base_datos = self.cargar_base_datos()

        # Separar características y etiquetas
        X = base_datos.drop(columns=["Archivo", "Etiqueta"]).values
        y = base_datos["Etiqueta"].values

        # Dividir la base de datos en n_splits particiones
        puntajes_split = []
        for split in range(self.n_splits):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / self.n_splits, random_state=split)

            # Usar Knn para entrenar y evaluar
            knn.X = X_train
            knn.y = y_train
            knn.candidato_X = X_test  # En este caso, X_test actúa como candidato temporal
            predicciones = [knn.clasificar() for _ in range(len(X_test))]
            puntaje_split = sum([1 for pred, real in zip(predicciones, y_test) if pred == real]) / len(y_test)
            puntajes_split.append(puntaje_split)

        # Calcular promedio de precisión en los splits
        promedio_puntaje = sum(puntajes_split) / self.n_splits
        print(f"Puntaje promedio para k = {k}: {promedio_puntaje:.4f}")
        return promedio_puntaje

    def optimizar_k(self):
        """
        Encuentra el valor de k que maximiza el puntaje promedio con validación cruzada.
        """
        mejor_k = None
        mejor_puntaje = 0

        for k in range(self.k_min, self.k_max + 1):
            puntaje_k = self.evaluar_con_validacion_cruzada(k)
            self.resultados_k[k] = puntaje_k

            if puntaje_k > mejor_puntaje:
                mejor_puntaje = puntaje_k
                mejor_k = k

        # Mostrar resultados finales
        print("\n=== Resultados de Optimización ===")
        print(f"Mejor valor de k: {mejor_k} con puntaje promedio: {mejor_puntaje:.4f}")
        return mejor_k, mejor_puntaje


# Punto de entrada para el programa
if __name__ == "__main__":
    puntuador = Puntuador(k_min=3, k_max=25, n_splits=5)
    puntuador.optimizar_k()
