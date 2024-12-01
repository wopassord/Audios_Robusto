import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class Knn:
    def __init__(self, k=3, n_componentes_pca=3):
        # Inicialización de rutas
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(self.base_path, "DB")
        self.csv_db_path = os.path.join(self.db_path, "caracteristicas.csv")
        self.csv_candidato_path = os.path.join(self.db_path, "Candidato", "candidato_caracteristicas.csv")

        # Parámetros de k-NN y PCA
        self.k = k
        self.n_componentes_pca = n_componentes_pca
        self.pca = None
        self.knn_model = None

    def _cargar_caracteristicas(self, file_path):
        # Cargar características desde un archivo CSV
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"No se encontró el archivo {file_path}")

        data = pd.read_csv(file_path, encoding='latin-1')
        return data

    def entrenar_modelo(self):
        # Entrenar el modelo k-NN con la base de datos
        data = self._cargar_caracteristicas(self.csv_db_path)

        # Extraer características (X) y etiquetas (y)
        X = data.iloc[:, 1:].values  # Todas las columnas excepto el nombre del archivo
        y = data["Archivo"].apply(lambda x: x.split("_")[0]).values  # Asumimos que la palabra está en el nombre del archivo

        # Aplicar PCA para reducir dimensionalidad a 3 componentes
        self.pca = PCA(n_components=self.n_componentes_pca)
        X_reducido = self.pca.fit_transform(X)

        # Crear y entrenar el modelo k-NN
        self.knn_model = KNeighborsClassifier(n_neighbors=self.k)
        self.knn_model.fit(X_reducido, y)
        print(f"Modelo entrenado con {len(data)} muestras y {self.n_componentes_pca} componentes principales.")

        # Graficar los puntos en 3D
        self._graficar_pca(X_reducido, y)

    def _graficar_pca(self, X_reducido, y):
        # Crear un gráfico 3D de los puntos reducidos
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Asignar colores únicos a cada clase
        clases = np.unique(y)
        colores = plt.cm.get_cmap("tab10", len(clases))

        for i, clase in enumerate(clases):
            mask = (y == clase)
            ax.scatter(X_reducido[mask, 0], X_reducido[mask, 1], X_reducido[mask, 2],
                       label=clase, color=colores(i), s=50)

        ax.set_title("Puntos en el Espacio de Componentes Principales (PCA)")
        ax.set_xlabel("Componente Principal 1")
        ax.set_ylabel("Componente Principal 2")
        ax.set_zlabel("Componente Principal 3")
        ax.legend()
        plt.show()

    def predecir_audio_candidato(self):
        # Predecir la palabra del audio candidato
        if self.knn_model is None or self.pca is None:
            raise ValueError("El modelo k-NN o PCA no han sido entrenados. Llame a entrenar_modelo primero.")

        candidato_data = self._cargar_caracteristicas(self.csv_candidato_path)
        X_candidato = candidato_data.iloc[:, 1:].values  # Todas las columnas excepto el nombre del archivo

        # Reducir dimensionalidad con PCA
        X_candidato_reducido = self.pca.transform(X_candidato)

        # Realizar la predicción
        prediccion = self.knn_model.predict(X_candidato_reducido)
        print(f"El audio candidato ha sido clasificado como: {prediccion[0]}")
        return prediccion[0]
