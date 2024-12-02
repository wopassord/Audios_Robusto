import os
import pandas as pd
import numpy as np
from Knn import Knn

class AnalizadorErrores:
    def __init__(self, base_datos_csv, candidato_csv):
        """
        Clase para analizar errores en el modelo k-NN.
        :param base_datos_csv: Ruta al archivo CSV de la base de datos aumentada.
        :param candidato_csv: Ruta al archivo CSV del audio candidato.
        """
        self.base_datos_csv = base_datos_csv
        self.candidato_csv = candidato_csv

    def analizar_errores_por_rango(self, k_min=2, k_max=20):
        """
        Analiza errores para un rango de valores de k.
        :param k_min: Valor mínimo de k.
        :param k_max: Valor máximo de k.
        """
        # Cargar datos
        base_datos = pd.read_csv(self.base_datos_csv)
        X = base_datos.drop(columns=["Archivo", "Etiqueta"]).values
        y_true = base_datos["Etiqueta"].values

        # Inicializar resultados
        resultados = []

        # Iterar sobre el rango de k
        for k in range(k_min, k_max + 1):
            print(f"Analizando errores para k = {k}...")
            knn = Knn(candidato_csv=self.candidato_csv, k=k)
            errores = []
            aciertos = 0

            # Iterar sobre cada punto de la base de datos como candidato
            for i in range(len(X)):
                knn.X = np.delete(X, i, axis=0)  # Base de datos sin el punto actual
                knn.y = np.delete(y_true, i)     # Etiquetas sin el punto actual
                knn.candidato_X = X[i].reshape(1, -1)  # Punto actual como candidato
                prediccion = knn.clasificar()

                if prediccion == y_true[i]:
                    aciertos += 1
                else:
                    errores.append({
                        "Archivo": base_datos.iloc[i]["Archivo"],
                        "Etiqueta Verdadera": y_true[i],
                        "Predicción": prediccion
                    })

            # Calcular precisión
            precision = aciertos / len(X)
            print(f"Precisión para k = {k}: {precision:.4f}")

            # Guardar resultados
            resultados.append({
                "k": k,
                "Precisión": precision,
                "Errores": len(errores),
                "Detalles Errores": errores
            })

        # Imprimir resumen
        print("\n=== Resumen de Resultados ===")
        for res in resultados:
            print(f"k = {res['k']}: Precisión = {res['Precisión']:.4f}, Errores = {res['Errores']}")

        return resultados

if __name__ == "__main__":
    base_datos_csv = r"C:\Users\berni\Desktop\ProyectoIAaudiosRobusto\Parametros\base_datos_aumentada_parametros.csv"
    candidato_csv = r"C:\Users\berni\Desktop\ProyectoIAaudiosRobusto\Parametros\candidato_parametros.csv"

    analizador = AnalizadorErrores(base_datos_csv, candidato_csv)
    resultados = analizador.analizar_errores_por_rango(k_min=2, k_max=20)

    # Opcional: Guardar resultados en un archivo CSV
    resumen = pd.DataFrame([{"k": r["k"], "Precisión": r["Precisión"], "Errores": r["Errores"]} for r in resultados])
    resumen.to_csv("resumen_errores.csv", index=False)
    print("Resultados guardados en 'resumen_errores.csv'")
