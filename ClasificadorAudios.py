from ProcesadorAudios import ProcesadorAudios
from Parametrizador import Parametrizador
from Knn import Knn
import os

class ClasificadorAudios:
    def __init__(self):
        # Inicializar las clases necesarias
        self.procesador = ProcesadorAudios()
        self.parametrizador = Parametrizador()
        candidato_csv = "DB/Candidato/candidato_parametros.csv"  # Ruta fija al archivo del candidato
        self.knn = Knn(candidato_csv=candidato_csv)  # Ajustar inicialización del Knn

    def mostrar_menu(self):
        # Mostrar el menú interactivo
        print("\n=== Clasificador de Audios ===")
        print("1. Preprocesar audios de la base de datos")
        print("2. Preprocesar audio candidato")
        print("3. Extraer características de la base de datos")
        print("4. Extraer características del audio candidato")
        print("5. Clasificar audio candidato (k-NN con PCA)")
        print("6. Clasificar audio candidato (k-NN sin PCA)")
        print("7. Salir")

    def ejecutar_opcion(self, opcion):
        # Ejecutar la opción seleccionada
        if opcion == "1":
            print("Preprocesando audios de la base de datos...")
            self.procesador.procesar_base_datos()
        elif opcion == "2":
            print("Preprocesando audio candidato...")
            candidato_path = input("Ingrese la ruta del audio candidato: ").strip()
            self.procesador.procesar_audio_candidato(candidato_path)
        elif opcion == "3":
            print("Extrayendo características de la base de datos...")
            self.parametrizador.procesar_base_datos()
        elif opcion == "4":
            print("Extrayendo características del audio candidato...")
            self.parametrizador.procesar_audio_candidato()
        elif opcion == "5":
            print("Clasificando audio candidato (k-NN con PCA)...")
            if not os.path.exists(self.knn.base_datos_csv):
                print(f"Error: El archivo {self.knn.base_datos_csv} no existe. Procesa primero la base de datos.")
            elif not os.path.exists(self.knn.candidato_csv):
                print(f"Error: El archivo {self.knn.candidato_csv} no existe. Procesa primero el audio candidato.")
            else:
                self.knn.usar_pca = True
                self.knn.cargar_datos()
                self.knn.visualizar_datos()
                prediccion = self.knn.clasificar()
                print(f"Predicción del audio candidato: {prediccion}")
        elif opcion == "6":
            print("Clasificando audio candidato (k-NN sin PCA)...")
            self.knn.usar_pca = False
            self.knn.cargar_datos()
            prediccion = self.knn.clasificar()
            print(f"Predicción del audio candidato: {prediccion}")
        elif opcion == "7":
            print("Saliendo del programa. ¡Hasta luego!")
            return False
        else:
            print("Opción no válida. Por favor, intente nuevamente.")
        return True

    def iniciar(self):
        # Loop principal del menú
        continuar = True
        while continuar:
            self.mostrar_menu()
            opcion = input("Seleccione una opción: ").strip()
            continuar = self.ejecutar_opcion(opcion)


# Punto de entrada para el programa
if __name__ == "__main__":
    clasificador = ClasificadorAudios()
    clasificador.iniciar()
