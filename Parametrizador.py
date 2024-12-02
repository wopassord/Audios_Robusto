import os
import pandas as pd
import librosa
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler

class Parametrizador:
    def __init__(self):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(self.base_path, "DB")
        self.processed_path = os.path.join(self.db_path, "Processed")
        self.augmented_path = os.path.join(self.db_path, "Augmented")
        self.parametros_path = os.path.join(self.base_path, "Parametros")
        self.candidato_path = os.path.join(self.db_path, "Candidato")
        os.makedirs(self.parametros_path, exist_ok=True)
        self.scaler = StandardScaler()

    def _band_pass_filter(self, data, sr, low, high):
        """
        Aplica un filtro pasa banda al audio.
        """
        nyquist = 0.5 * sr
        low = low / nyquist
        high = high / nyquist
        b, a = butter(4, [low, high], btype="band")
        return lfilter(b, a, data)

    def _extraer_caracteristicas(self, audio, sr):
        """
        Extrae características como MFCCs, ZCR, espectro, y bandas de energía.
        """
        # MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Zero-Crossing Rate (ZCR)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))

        # Centroide Espectral y Ancho Espectral
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))

        # Energía en Bandas Específicas
        bands = [(300, 800), (800, 1600), (1600, 3400)]
        energies = []
        for low, high in bands:
            filtered_audio = self._band_pass_filter(audio, sr=sr, low=low, high=high)
            energies.append(np.sum(filtered_audio ** 2))

        # Consolidar Características
        features = list(mfcc_mean) + [zcr, spectral_centroid, spectral_bandwidth] + energies
        return features

    def _procesar_y_guardar(self, folder_path, output_file):
        """
        Procesa todos los audios en una carpeta, extrae sus características, 
        les asigna etiquetas y las guarda en un CSV.
        """
        registros = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(folder_path, file_name)
                audio, sr = librosa.load(file_path, sr=None)
                caracteristicas = self._extraer_caracteristicas(audio, sr)
                
                # Asignar etiquetas según nombres de archivos
                if "zanahoria" in file_name.lower():
                    etiqueta = "zanahoria"
                elif "papa" in file_name.lower():
                    etiqueta = "papa"
                elif "camote" in file_name.lower():
                    etiqueta = "camote"
                elif "berenjena" in file_name.lower():
                    etiqueta = "berenjena"
                else:
                    etiqueta = "desconocido"

                # Agregar información al registro
                caracteristicas.append(file_name)
                caracteristicas.append(etiqueta)
                registros.append(caracteristicas)

        # Crear DataFrame y Escalar Características
        columnas = (
            [f"MFCC_{i}" for i in range(1, 14)]
            + ["ZCR", "CentroideEspectral", "AnchoEspectral", "EnergiaBanda_1", "EnergiaBanda_2", "EnergiaBanda_3", "Archivo", "Etiqueta"]
        )
        df = pd.DataFrame(registros, columns=columnas)

        # Escalar Características Numéricas (Excluyendo "Archivo" y "Etiqueta")
        caracteristicas_num = df.drop(columns=["Archivo", "Etiqueta"])
        df_scaled = pd.DataFrame(self.scaler.fit_transform(caracteristicas_num), columns=caracteristicas_num.columns)
        df_scaled["Archivo"] = df["Archivo"]
        df_scaled["Etiqueta"] = df["Etiqueta"]

        # Guardar Características en CSV
        output_path = os.path.join(self.parametros_path, output_file)
        df_scaled.to_csv(output_path, index=False)
        print(f"Características guardadas en {output_path}")

    def procesar_base_datos(self):
        """
        Procesa los audios de la base de datos (procesados y aumentados) y guarda sus características en un CSV.
        """
        self._procesar_y_guardar(self.processed_path, "base_datos_parametros.csv")
        self._procesar_y_guardar(self.augmented_path, "base_datos_aumentada_parametros.csv")

    def procesar_audio_candidato(self):
        """
        Procesa únicamente el archivo `candidato_procesado.wav` y guarda sus características en un CSV,
        aplicando el mismo escalado utilizado para la base de datos aumentada.
        """
        # Ruta del archivo esperado
        candidato_file = "candidato_procesado.wav"
        candidato_path = os.path.join(self.candidato_path, candidato_file)

        # Verificar si el archivo existe
        if not os.path.exists(candidato_path):
            raise FileNotFoundError(f"El archivo {candidato_file} no se encontró en la carpeta {self.candidato_path}.")

        # Procesar el archivo
        registros = []
        audio, sr = librosa.load(candidato_path, sr=None)
        caracteristicas = self._extraer_caracteristicas(audio, sr)
        caracteristicas.append(candidato_file)
        registros.append(caracteristicas)

        # Crear DataFrame
        columnas = [f"MFCC_{i}" for i in range(1, 14)] + ["ZCR", "CentroideEspectral", "AnchoEspectral",
                                                        "EnergiaBanda_1", "EnergiaBanda_2", "EnergiaBanda_3", "Archivo"]
        df = pd.DataFrame(registros, columns=columnas)

        # Verificar si el escalador está ajustado con la base de datos aumentada
        base_datos_aumentada_path = os.path.join(self.parametros_path, "base_datos_aumentada_parametros.csv")
        if not hasattr(self.scaler, "mean_"):
            print("Ajustando el escalador con los datos de la base de datos aumentada...")
            if not os.path.exists(base_datos_aumentada_path):
                raise FileNotFoundError(f"La base de datos aumentada no se encontró en {base_datos_aumentada_path}.")
            base_datos_aumentada = pd.read_csv(base_datos_aumentada_path)
            caracteristicas_base = base_datos_aumentada.drop(columns=["Archivo", "Etiqueta"])
            self.scaler.fit(caracteristicas_base)

        # **Escalar Características del Audio Candidato**
        caracteristicas_num = df.drop(columns=["Archivo"])
        try:
            df_scaled = pd.DataFrame(self.scaler.transform(caracteristicas_num), columns=caracteristicas_num.columns)
        except ValueError as e:
            print(f"Error al normalizar las características: {e}")
            raise

        # Agregar la columna del archivo al DataFrame escalado
        df_scaled["Archivo"] = df["Archivo"]

        # Guardar Características en CSV (sobrescribir siempre)
        output_path = os.path.join(self.parametros_path, "candidato_parametros.csv")
        df_scaled.to_csv(output_path, index=False)
        print(f"Características del audio candidato normalizadas y guardadas en {output_path}")
