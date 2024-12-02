import os
import pandas as pd
import numpy as np
from Knn import Knn

class AnalizadorErrores:
    def __init__(self, k=4):
        """
        Clase para analizar los errores del modelo k-NN.
        :param k: Valor de k para k-NN.
        """
        self.k = k
        self.parametros_db_path = r"C:\Users\berni\Desktop\ProyectoIAaudiosRobusto\Parametros\base_datos_aumentada_parametros.csv"
        self.knn = Knn(candidato_csv=None, k=k)

    def cargar_base_datos(self):
        """
        Carga la base de datos aumentada.
        """
        if not os.path.exists(self.parametros_db_path):
            raise FileNotFoundError(f"La base de datos en {self.parametros_db_path} no existe. Procesa primero los audios.")
        return pd.read_csv(self.parametros_db_path)

    def analizar_errores(self):
        """
        Analiza los errores cometidos por el modelo y genera estadísticas detalladas.
        """
        # Cargar la base de datos aumentada
        base_datos = self.cargar_base_datos()

        # Inicializar contadores
        total = len(base_datos)
        errores = 0
        resumen_errores = []

        # Iterar sobre cada audio de la base de datos
        for idx in range(total):
            # Crear conjuntos de entrenamiento y validación
            base_entrenamiento = base_datos.drop(idx).reset_index(drop=True)
            audio_candidato = base_datos.iloc[idx]

            # Cargar datos en el modelo k-NN
            self.knn.X = base_entrenamiento.drop(columns=["Archivo", "Etiqueta"]).values
            self.knn.y = base_entrenamiento["Etiqueta"].values
            self.knn.candidato_X = audio_candidato.drop(["Archivo", "Etiqueta"]).values.reshape(1, -1)

            # Clasificar el audio candidato
            etiqueta_real = audio_candidato["Etiqueta"]
            prediccion = self.knn.clasificar()

            # Registrar error si aplica
            if prediccion != etiqueta_real:
                errores += 1
                resumen_errores.append({
                    "Archivo": audio_candidato["Archivo"],
                    "Etiqueta Real": etiqueta_real,
                    "Predicción": prediccion
                })

        # Calcular precisión
        precision = (total - errores) / total
        print(f"\n=== Análisis de Errores para k = {self.k} ===")
        print(f"Total de audios: {total}")
        print(f"Errores: {errores}")
        print(f"Precisión: {precision:.4f}")

        # Guardar resumen de errores en un CSV
        errores_df = pd.DataFrame(resumen_errores)
        errores_path = os.path.join(os.path.dirname(self.parametros_db_path), f"resumen_errores_k_{self.k}.csv")
        errores_df.to_csv(errores_path, index=False)
        print(f"Resumen de errores guardado en: {errores_path}")

        # Generar estadísticas de errores
        self.generar_estadisticas_errores(resumen_errores)

    def generar_estadisticas_errores(self, resumen_errores):
        """
        Genera estadísticas detalladas sobre las clases más confundidas.
        """
        errores_df = pd.DataFrame(resumen_errores)
        if errores_df.empty:
            print("No se encontraron errores.")
            return

        # Contar errores por etiqueta real y predicción
        confusion = errores_df.groupby(["Etiqueta Real", "Predicción"]).size().reset_index(name="Frecuencia")
        print("\n=== Confusiones Más Frecuentes ===")
        print(confusion.sort_values(by="Frecuencia", ascending=False).head())

        # Identificar las clases más problemáticas
        clases_problematicas = confusion.groupby("Etiqueta Real")["Frecuencia"].sum().reset_index()
        clases_problematicas = clases_problematicas.sort_values(by="Frecuencia", ascending=False)

        print("\n=== Clases con Más Errores ===")
        print(clases_problematicas.head())

# Punto de entrada para el programa
if __name__ == "__main__":
    analizador = AnalizadorErrores(k=3)
    analizador.analizar_errores()
