import os
import librosa
import numpy as np
import shutil
import soundfile as sf
from scipy.signal import butter, lfilter


class ProcesadorAudios:
    def __init__(self):
        # Rutas de carpetas
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(self.base_path, "DB")
        self.crudos_path = os.path.join(self.db_path, "Crudos")
        self.processed_path = os.path.join(self.db_path, "Processed")
        self.candidato_path = os.path.join(self.db_path, "Candidato")
        self.sampling_rate = 16000  # Frecuencia de muestreo estándar para audios
        self.target_duration = 2.0  # Duración uniforme en segundos para los audios procesados

        # Crear carpetas si no existen
        os.makedirs(self.processed_path, exist_ok=True)
        os.makedirs(self.candidato_path, exist_ok=True)

    def _butter_bandpass_filter(self, data, lowcut=300.0, highcut=3400.0, fs=16000, order=5):
        # Implementación del filtro pasa banda
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    def _normalize_audio(self, audio):
        # Normalizar amplitud a rango [-1, 1]
        return audio / np.max(np.abs(audio))

    def _remove_silence(self, audio, sr, threshold_db=-25):
        # Remover silencios del audio
        intervals = librosa.effects.split(audio, top_db=threshold_db)
        if len(intervals) == 0:  # Si no se detectaron intervalos
            print("Advertencia: No se detectaron intervalos de audio que superen el umbral de silencio.")
            return audio  # Devolver el audio original
        return np.concatenate([audio[start:end] for start, end in intervals])

    def _adjust_duration(self, audio):
        # Ajustar la duración del audio a target_duration (truncar o rellenar con ceros)
        target_length = int(self.target_duration * self.sampling_rate)
        if len(audio) > target_length:
            return audio[:target_length]
        else:
            return np.pad(audio, (0, target_length - len(audio)))

    def _calculate_band_energy(self, audio, sr):
        # Calcular energía en bandas de frecuencia específicas
        bands = [(300, 800), (800, 1600), (1600, 3400)]
        energies = []
        for low, high in bands:
            band_filtered = self._butter_bandpass_filter(audio, lowcut=low, highcut=high, fs=sr)
            energies.append(np.sum(band_filtered ** 2))
        return energies

    def _process_audio(self, file_path):
        # Procesar un solo archivo de audio
        audio, sr = librosa.load(file_path, sr=self.sampling_rate)

        # Pasos de preprocesamiento
        audio = self._butter_bandpass_filter(audio, fs=sr)
        audio = self._remove_silence(audio, sr)
        audio = self._normalize_audio(audio)
        audio = self._adjust_duration(audio)

        return audio

    def procesar_base_datos(self):
        # Procesar todos los audios en la carpeta Crudos
        for file_name in os.listdir(self.crudos_path):
            if file_name.endswith((".wav", ".mp3")):
                file_path = os.path.join(self.crudos_path, file_name)
                processed_audio = self._process_audio(file_path)

                # Guardar el audio procesado
                output_path = os.path.join(self.processed_path, file_name.replace(".mp3", ".wav"))
                sf.write(output_path, processed_audio, self.sampling_rate)  # Usando soundfile
                print(f"Audio procesado y guardado: {output_path}")

    def procesar_audio_candidato(self, audio_candidato_path):
        # Normalizar la ruta proporcionada
        audio_candidato_path = os.path.normcase(os.path.normpath(audio_candidato_path.strip()))

        # Verificar si el archivo existe
        if not os.path.isfile(audio_candidato_path):
            raise FileNotFoundError(f"El archivo {audio_candidato_path} no existe o no es accesible.")

        # Normalizar la ruta de destino en la carpeta Candidato
        candidato_name = os.path.basename(audio_candidato_path)
        candidato_path = os.path.normcase(os.path.normpath(os.path.join(self.candidato_path, candidato_name)))

        # Verificar si el archivo ya está en la carpeta Candidato
        if audio_candidato_path != candidato_path:  # Solo copiar si no están en la misma ubicación
            shutil.copy(audio_candidato_path, self.candidato_path)

        # Procesar el audio
        processed_audio = self._process_audio(candidato_path)

        # Guardar el audio procesado
        output_path = os.path.join(self.candidato_path, "candidato_procesado.wav")
        sf.write(output_path, processed_audio, self.sampling_rate)  # Usando soundfile
        print(f"Audio candidato procesado y guardado: {output_path}")
