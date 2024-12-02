import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif


class AnalizadorParametros:
    def __init__(self, base_datos_csv):
        """
        Clase para analizar la relevancia de los parámetros de los datos de audio.
        :param base_datos_csv: Ruta al archivo CSV con la base de datos aumentada.
        """
        self.base_datos_csv = base_datos_csv
        self.base_datos = None
        self.X = None
        self.y = None

    def cargar_datos(self):
        """
        Carga la base de datos y separa las características y las etiquetas.
        """
        self.base_datos = pd.read_csv(self.base_datos_csv)
        self.X = self.base_datos.drop(columns=["Archivo", "Etiqueta"])
        self.y = self.base_datos["Etiqueta"]

    def correlacion_entre_parametros(self):
        """
        Calcula la matriz de correlación entre los parámetros y la visualiza.
        """
        print("Calculando matriz de correlación entre parámetros...")
        correlacion = self.X.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlacion, annot=False, cmap="coolwarm", cbar=True)
        plt.title("Matriz de Correlación entre Parámetros")
        plt.show()

    def correlacion_parametros_clases(self):
        """
        Calcula la correlación de los parámetros con las etiquetas.
        """
        print("Calculando correlación entre parámetros y etiquetas...")
        correlaciones = {}
        etiquetas = self.y.unique()

        for columna in self.X.columns:
            correlacion_promedio = 0
            for etiqueta in etiquetas:
                correlacion_promedio += self.X[self.y == etiqueta][columna].mean()
            correlacion_promedio /= len(etiquetas)
            correlaciones[columna] = correlacion_promedio

        correlaciones_ordenadas = sorted(correlaciones.items(), key=lambda x: abs(x[1]), reverse=True)
        print("\nCorrelación de parámetros con etiquetas (promedio):")
        for parametro, correlacion in correlaciones_ordenadas:
            print(f"{parametro}: {correlacion:.4f}")

    def importancia_por_random_forest(self):
        """
        Calcula la importancia de cada parámetro utilizando Random Forest.
        """
        print("Calculando importancia de parámetros usando Random Forest...")
        modelo = RandomForestClassifier(random_state=42)
        modelo.fit(self.X, self.y)

        importancia = modelo.feature_importances_
        importancia_df = pd.DataFrame({
            "Parámetro": self.X.columns,
            "Importancia": importancia
        }).sort_values(by="Importancia", ascending=False)

        print("\nImportancia de parámetros (Random Forest):")
        print(importancia_df)

        plt.figure(figsize=(12, 6))
        sns.barplot(x="Importancia", y="Parámetro", data=importancia_df)
        plt.title("Importancia de Parámetros según Random Forest")
        plt.show()

    def seleccion_mejores_parametros(self, k=10):
        """
        Selecciona los mejores parámetros basados en ANOVA F-value.
        :param k: Número de parámetros a seleccionar.
        """
        print(f"Seleccionando los {k} mejores parámetros utilizando ANOVA F-value...")
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(self.X, self.y)

        seleccionados = self.X.columns[selector.get_support()]
        print(f"\nLos {k} mejores parámetros seleccionados son:")
        for parametro in seleccionados:
            print(f"- {parametro}")

        return seleccionados


# Punto de entrada para analizar los parámetros
if __name__ == "__main__":
    base_datos_csv = r"C:\Users\berni\Desktop\ProyectoIAaudiosRobusto\Parametros\base_datos_aumentada_parametros.csv"
    analizador = AnalizadorParametros(base_datos_csv)
    analizador.cargar_datos()

    # Matriz de correlación entre parámetros
    analizador.correlacion_entre_parametros()

    # Correlación entre parámetros y clases
    analizador.correlacion_parametros_clases()

    # Importancia según Random Forest
    analizador.importancia_por_random_forest()

    # Selección de los mejores parámetros
    analizador.seleccion_mejores_parametros(k=10)
