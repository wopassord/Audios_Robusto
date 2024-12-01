import os
import librosa
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter


class Parametrizador:
    def __init__(self):
        # Rutas de carpetas
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(self.base_path, "DB")
        self.processed_path = os.path.join(self.db_path, "Processed")
        self.candidato_path = os.path.join(self.db_path, "Candidato")
        self.output_csv = os.path.join(self.processed_path, "caracteristicas.csv")
        self.candidato_csv = os.path.join(self.candidato_path, "candidato_caracteristicas.csv")

    def _calculate_band_energy(self, audio, sr):
        # Calcular energía en bandas de frecuencia específicas
        bands = [(300, 800), (800, 1600), (1600, 3400)]
        energies = []
        for low, high in bands:
            nyquist = 0.5 * sr
            low_norm = low / nyquist
            high_norm = high / nyquist
            b, a = butter(5, [low_norm, high_norm], btype='band')
            band_filtered = lfilter(b, a, audio)
            energies.append(np.sum(band_filtered ** 2))
        return energies

    def _extraer_caracteristicas(self, audio, sr):
        try:
            print("Iniciando extracción de características...")

            # MFCC y Delta MFCC
            raw_mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            print(f"MFCC shape: {raw_mfccs.shape}")
            mfccs = np.mean(raw_mfccs.T, axis=0)

            raw_delta_mfccs = librosa.feature.delta(raw_mfccs)
            print(f"Delta MFCC shape: {raw_delta_mfccs.shape}")
            delta_mfccs = np.mean(raw_delta_mfccs.T, axis=0)

            # Contraste Espectral
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            print(f"Contraste Espectral shape: {contrast.shape}")
            contrast = np.mean(contrast.T, axis=0)

            # Centroide Espectral y Ancho Espectral
            centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            print(f"Centroide Espectral: {centroid}")

            bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
            print(f"Ancho Espectral: {bandwidth}")

            # Zero Crossing Rate y Energía Total
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
            print(f"ZCR: {zcr}")

            energia_total = np.sum(audio ** 2)
            print(f"Energía Total: {energia_total}")

            # Energías de Bandas
            band_energies = self._calculate_band_energy(audio, sr)
            print(f"Energías de Bandas: {band_energies}")

            # Validación de Dimensiones
            if len(band_energies) != 3:
                raise ValueError(f"El cálculo de Energía de Bandas no generó 3 valores, obtenidos: {len(band_energies)}.")

            # Concatenar todas las características
            features = np.hstack([
                mfccs,              # 13 características
                delta_mfccs,        # 13 características
                contrast,           # 7 características
                [centroid],         # 1 característica
                [bandwidth],        # 1 característica
                [zcr],              # 1 característica
                [energia_total],    # 1 característica
                band_energies       # 3 características
            ])

            # Debug Final
            print(f"Features concatenados: {features}")
            print(f"Shape final de features: {features.shape}")

            # Validar Shape Final
            if features.shape[0] != 41:
                raise ValueError(f"Se esperaban 41 características, pero se obtuvieron {features.shape[0]}.")

            return features
        except Exception as e:
            print(f"Error en _extraer_caracteristicas: {e}")
            raise

    def _procesar_audio(self, file_path):
        # Cargar audio procesado
        audio, sr = librosa.load(file_path, sr=None)
        return self._extraer_caracteristicas(audio, sr)

    def parametrizar_base_datos(self):
        # Extraer características de todos los audios procesados
        caracteristicas = []
        for file_name in os.listdir(self.processed_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(self.processed_path, file_name)
                features = self._procesar_audio(file_path)
                if len(features) != 41:  # Validar la longitud de las características
                    raise ValueError(f"El audio {file_name} tiene {len(features)} características, pero se esperaban 41.")
                caracteristicas.append([file_name] + features.tolist())

        # Guardar características en un archivo CSV
        header = ["Archivo"] + [f"MFCC{i+1}" for i in range(13)] + \
                [f"Delta_MFCC{i+1}" for i in range(13)] + \
                [f"Contraste{i+1}" for i in range(7)] + \
                ["Centroide", "Ancho", "ZCR", "Energia_Total"] + \
                [f"Energia_Banda{i+1}" for i in range(3)]
        df = pd.DataFrame(caracteristicas, columns=header)
        df.to_csv(self.output_csv, index=False)
        print(f"Características de la base de datos guardadas en: {self.output_csv}")

    def parametrizar_audio_candidato(self):
        # Extraer características del audio candidato procesado
        candidato_path = os.path.join(self.candidato_path, "candidato_procesado.wav")
        if not os.path.isfile(candidato_path):
            raise FileNotFoundError(f"El archivo {candidato_path} no existe. Procese el audio candidato primero.")

        features = self._procesar_audio(candidato_path)
        header = ["Archivo"] + [f"MFCC{i+1}" for i in range(13)] + \
                 [f"Delta_MFCC{i+1}" for i in range(13)] + \
                 [f"Contraste{i+1}" for i in range(7)] + \
                 ["Centroide", "Ancho", "ZCR", "Energia_Total"] + \
                 [f"Energia_Banda{i+1}" for i in range(3)]
        df = pd.DataFrame([[os.path.basename(candidato_path)] + features.tolist()], columns=header)
        df.to_csv(self.candidato_csv, index=False)
        print(f"Características del audio candidato guardadas en: {self.candidato_csv}")
