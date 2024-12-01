from ProcesadorAudios import ProcesadorAudios
from Parametrizador import Parametrizador
from Knn import Knn

class ClasificadorAudios:
    def __init__(self):
        # Inicializar las clases necesarias
        self.procesador = ProcesadorAudios()
        self.parametrizador = Parametrizador()
        self.knn = Knn(k=3, n_componentes_pca=3)  # Ajuste de k y PCA

    def mostrar_menu(self):
        # Mostrar el menú interactivo
        print("\n=== Clasificador de Audios ===")
        print("1. Preprocesar audios de la base de datos")
        print("2. Preprocesar audio candidato")
        print("3. Extraer características de la base de datos")
        print("4. Extraer características del audio candidato")
        print("5. Entrenar modelo k-NN (y graficar en 3D)")
        print("6. Clasificar audio candidato")
        print("7. Mostrar espectrogramas generados")
        print("8. Salir")

    def ejecutar_opcion(self, opcion):
        # Ejecutar la opción seleccionada
        if opcion == "1":
            print("Preprocesando audios de la base de datos...")
            self.procesador.procesar_base_datos()
            print("Cálculo de energías de bandas y espectrogramas completados.")
        elif opcion == "2":
            print("Preprocesando audio candidato...")
            candidato_path = input("Ingrese la ruta del audio candidato: ").strip()
            self.procesador.procesar_audio_candidato(candidato_path)
        elif opcion == "3":
            print("Extrayendo características de la base de datos...")
            self.parametrizador.parametrizar_base_datos()
        elif opcion == "4":
            print("Extrayendo características del audio candidato...")
            self.parametrizador.parametrizar_audio_candidato()
        elif opcion == "5":
            print("Entrenando modelo k-NN y graficando en 3D...")
            self.knn.entrenar_modelo()
        elif opcion == "6":
            print("Clasificando audio candidato...")
            self.knn.predecir_audio_candidato()
        elif opcion == "7":
            print("Espectrogramas generados durante el preprocesamiento se encuentran en la carpeta 'Processed'.")
        elif opcion == "8":
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
