import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import os

class Knn:
    def __init__(self, candidato_csv, k=5, usar_pca=True, excluir_parametros=None, pesos='distance', correlation_threshold=0.95):
        """
        Clase para implementar el modelo k-NN con exclusión automática de parámetros redundantes.
        :param candidato_csv: Ruta al archivo CSV del audio candidato.
        :param k: Número de vecinos para k-NN.
        :param usar_pca: Si True, aplica PCA con 3 componentes.
        :param excluir_parametros: Lista de nombres de columnas a excluir antes de procesar.
        :param pesos: Método para asignar peso a los vecinos ('uniform' o 'distance').
        :param correlation_threshold: Umbral de correlación para excluir parámetros redundantes (por defecto 0.95).
        """
        # Rutas y configuración
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_datos_csv = os.path.join(base_dir, "Parametros", "base_datos_aumentada_parametros.csv")
        self.candidato_csv = os.path.join(base_dir, "Parametros", "candidato_parametros.csv")
        self.k = k
        self.usar_pca = usar_pca
        self.excluir_parametros = excluir_parametros if excluir_parametros else []
        self.scaler = StandardScaler()
        self.pesos = pesos
        self.correlation_threshold = correlation_threshold

    def _seleccionar_parametros(self, data_frame):
        """
        Identifica y excluye parámetros redundantes basados en la matriz de correlación.
        :param data_frame: DataFrame con las características de la base de datos.
        :return: Lista de nombres de columnas seleccionadas.
        """
        # Calcular la matriz de correlación
        correlacion = data_frame.corr().abs()

        # Identificar columnas redundantes
        columnas_redundantes = set()
        for i in range(correlacion.shape[0]):
            for j in range(i + 1, correlacion.shape[1]):
                if correlacion.iloc[i, j] > self.correlation_threshold:
                    columnas_redundantes.add(correlacion.columns[j])

        # Excluir columnas redundantes
        columnas_seleccionadas = [col for col in data_frame.columns if col not in columnas_redundantes]
        print(f"Parámetros seleccionados: {columnas_seleccionadas}")
        return columnas_seleccionadas

    def cargar_datos(self):
        """
        Carga los datos de la base de datos aumentada y del candidato,
        excluyendo automáticamente parámetros redundantes según la matriz de correlación.
        """
        # Cargar la base de datos aumentada
        self.base_datos = pd.read_csv(self.base_datos_csv)
        self.candidato = pd.read_csv(self.candidato_csv)

        # Excluir columnas no numéricas o redundantes
        columnas_numericas = self.base_datos.select_dtypes(include=[np.number]).columns.tolist()
        columnas_a_usar = self._seleccionar_parametros(self.base_datos[columnas_numericas])
        columnas_a_excluir = set(self.base_datos.columns) - set(columnas_a_usar) - {"Etiqueta"}

        # Validar si las columnas a excluir existen antes de intentar eliminarlas
        columnas_a_excluir = [col for col in columnas_a_excluir if col in self.base_datos.columns]

        # Aplicar exclusión de parámetros
        self.X = self.base_datos.drop(columns=columnas_a_excluir, errors="ignore").drop(columns=["Etiqueta"], errors="ignore").values
        self.y = self.base_datos["Etiqueta"].values

        # Validar si las columnas a excluir existen antes de intentar eliminarlas en el candidato
        columnas_a_excluir_candidato = [col for col in columnas_a_excluir if col in self.candidato.columns]
        self.candidato_X = self.candidato.drop(columns=columnas_a_excluir_candidato, errors="ignore").values

        # Escalar las características
        self.X = self.scaler.fit_transform(self.X)
        self.candidato_X = self.scaler.transform(self.candidato_X)

    def aplicar_pca(self):
        """
        Aplica PCA para reducir las dimensiones a 3.
        """
        pca = PCA(n_components=3)
        self.X_pca = pca.fit_transform(self.X)
        self.candidato_X_pca = pca.transform(self.candidato_X)

    def optimizar_k(self):
        """
        Optimiza el valor de k utilizando validación cruzada.
        """
        k_valores = list(range(1, 21))
        k_scores = []

        for k in k_valores:
            knn = KNeighborsClassifier(n_neighbors=k, weights=self.pesos)
            scores = cross_val_score(knn, self.X, self.y, cv=5, scoring="accuracy")
            k_scores.append(scores.mean())

        # Determinar el mejor k
        self.k = k_valores[np.argmax(k_scores)]
        print(f"El valor óptimo de k es: {self.k}")

        # Graficar los resultados
        plt.plot(k_valores, k_scores)
        plt.xlabel("Número de vecinos (k)")
        plt.ylabel("Precisión promedio")
        plt.title("Optimización de k con validación cruzada")
        plt.show()

    def clasificar(self):
        """
        Clasifica el audio candidato usando k-NN.
        """
        if self.usar_pca:
            self.aplicar_pca()
            X = self.X_pca
            candidato_X = self.candidato_X_pca
        else:
            X = self.X
            candidato_X = self.candidato_X

        # Entrenar el modelo k-NN con pesos
        knn = KNeighborsClassifier(n_neighbors=self.k, weights=self.pesos)
        knn.fit(X, self.y)

        # Clasificar el candidato
        prediccion = knn.predict(candidato_X)
        print(f"El audio candidato fue clasificado como: {prediccion[0]}")

        return prediccion[0]


    def visualizar_datos(self):
        """
        Visualiza los datos en 3D si se usa PCA.
        """
        if not self.usar_pca:
            print("La visualización 3D solo está disponible con PCA habilitado.")
            return

        # Asegurar que se aplique PCA si no se ha hecho aún
        if not hasattr(self, "X_pca") or not hasattr(self, "candidato_X_pca"):
            self.aplicar_pca()

        # Validar la consistencia de las dimensiones entre `self.y` y `self.X_pca`
        if len(self.y) != len(self.X_pca):
            print(f"Error: Inconsistencia en dimensiones entre etiquetas ({len(self.y)}) y PCA ({len(self.X_pca)}).")
            return

        # Colores para las etiquetas
        colores = {
            "zanahoria": "orange",
            "papa": "yellow",
            "camote": "violet",
            "berenjena": "darkviolet",
            "desconocido": "gray"
        }

        # Crear gráfico 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Graficar la base de datos
        for etiqueta in np.unique(self.y):
            idx = np.where(self.y == etiqueta)[0]  # Asegurar que el índice booleano se mapea correctamente
            ax.scatter(
                self.X_pca[idx, 0], self.X_pca[idx, 1], self.X_pca[idx, 2],
                label=etiqueta, c=colores.get(etiqueta, "gray"), alpha=0.7
            )

        # Graficar el candidato en rojo
        ax.scatter(
            self.candidato_X_pca[:, 0], self.candidato_X_pca[:, 1], self.candidato_X_pca[:, 2],
            label="Candidato", c="red", marker="*", s=100
        )

        ax.set_title("Visualización PCA de la Base de Datos y el Candidato")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.legend()
        plt.show()


    def optimizar_k(self):
        """
        Optimiza el valor de k utilizando validación cruzada.
        """
        k_valores = list(range(1, 21))
        k_scores = []

        for k in k_valores:
            knn = KNeighborsClassifier(n_neighbors=k, weights=self.pesos)
            scores = cross_val_score(knn, self.X, self.y, cv=5, scoring="accuracy")
            k_scores.append(scores.mean())

        # Determinar el mejor k
        self.k = k_valores[np.argmax(k_scores)]
        print(f"El valor óptimo de k es: {self.k}")

        # Graficar los resultados
        plt.plot(k_valores, k_scores)
        plt.xlabel("Número de vecinos (k)")
        plt.ylabel("Precisión promedio")
        plt.title("Optimización de k con validación cruzada")
        plt.show()

    def clasificar(self):
        """
        Clasifica el audio candidato usando k-NN.
        """
        # Usar PCA si está activado
        if self.usar_pca:
            self.aplicar_pca()
            X = self.X_pca
            candidato_X = self.candidato_X_pca
        else:
            X = self.X
            candidato_X = self.candidato_X

        # Entrenar el modelo k-NN con pesos
        knn = KNeighborsClassifier(n_neighbors=self.k, weights=self.pesos)
        knn.fit(X, self.y)

        # Clasificar el candidato
        prediccion = knn.predict(candidato_X)
        print(f"El audio candidato fue clasificado como: {prediccion[0]}")

        return prediccion[0]
