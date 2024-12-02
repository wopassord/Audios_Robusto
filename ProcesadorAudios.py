import os
import librosa
import numpy as np
import shutil
import soundfile as sf
from scipy.signal import butter, lfilter

class ProcesadorAudios:
    def __init__(self):
            self.base_path = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(self.base_path, "DB")
            self.crudos_path = os.path.join(self.db_path, "Crudos")
            self.processed_path = os.path.join(self.db_path, "Processed")
            self.augmented_path = os.path.join(self.db_path, "Augmented")
            self.candidato_path = os.path.join(self.db_path, "Candidato")  # Definir la ruta del candidato
            self.sampling_rate = 16000
            self.target_duration = 2.0

            os.makedirs(self.processed_path, exist_ok=True)
            os.makedirs(self.augmented_path, exist_ok=True)
            os.makedirs(self.candidato_path, exist_ok=True)  # Crear carpeta de candidato si no existe


    def _butter_bandpass_filter(self, data, lowcut=300.0, highcut=3400.0, fs=16000, order=5):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='band')
            return lfilter(b, a, data)

    def _normalize_audio(self, audio):
        return audio / np.max(np.abs(audio))
    
    def _remove_silence(self, audio, sr, threshold_db=-25):
            intervals = librosa.effects.split(audio, top_db=threshold_db)
            if len(intervals) == 0:
                print("Advertencia: No se detectaron intervalos de audio que superen el umbral de silencio.")
                return audio
            return np.concatenate([audio[start:end] for start, end in intervals])

    def _adjust_duration(self, audio):
        target_length = int(self.target_duration * self.sampling_rate)
        if len(audio) > target_length:
            return audio[:target_length]
        else:
            return np.pad(audio, (0, target_length - len(audio)))
        
    def _augment_audio(self, audio, sr):
        """
        Realiza augmentación del audio original con las siguientes técnicas:
        - Cambio de tono (pitch shift)
        - Agregar ruido blanco
        - Cambiar velocidad ajustando la duración manualmente
        """
        augmented_audios = []

        # Cambio de tono
        for n_steps in [-2, 2]:  # Dos tonos arriba y abajo
            augmented_audios.append(librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps))

        # Agregar ruido
        noise = np.random.normal(0, 0.01, len(audio))
        augmented_audios.append(audio + noise)

        # Cambiar velocidad ajustando la duración manualmente
        for rate in [0.9, 1.1]:  # Más lento y más rápido
            # Aplicar `time_stretch` con el argumento `rate` explícito
            stretched_audio = librosa.effects.time_stretch(audio, rate=rate)
            adjusted_audio = self._adjust_duration(stretched_audio)
            augmented_audios.append(adjusted_audio)

        return augmented_audios


    def _process_audio(self, file_path):
        audio, sr = librosa.load(file_path, sr=self.sampling_rate)
        audio = self._butter_bandpass_filter(audio, fs=sr)
        audio = self._remove_silence(audio, sr)
        audio = self._normalize_audio(audio)
        audio = self._adjust_duration(audio)
        return audio

    def procesar_base_datos(self):
        for file_name in os.listdir(self.crudos_path):
            if file_name.endswith((".wav", ".mp3")):
                file_path = os.path.join(self.crudos_path, file_name)
                processed_audio = self._process_audio(file_path)

                output_path = os.path.join(self.processed_path, file_name.replace(".mp3", ".wav"))
                sf.write(output_path, processed_audio, self.sampling_rate)
                print(f"Audio procesado y guardado: {output_path}")

                # Realizar augmentación
                augmented_audios = self._augment_audio(processed_audio, self.sampling_rate)
                for i, augmented_audio in enumerate(augmented_audios):
                    aug_file_name = file_name.replace(".mp3", "").replace(".wav", f"_aug_{i + 1}.wav")
                    aug_output_path = os.path.join(self.augmented_path, aug_file_name)
                    sf.write(aug_output_path, augmented_audio, self.sampling_rate)
                    print(f"Audio aumentado guardado: {aug_output_path}")

    def procesar_audio_candidato(self, audio_candidato_path):
        """
        Procesa el audio candidato y lo guarda como `candidato_procesado.wav` en la carpeta `Candidato`.
        :param audio_candidato_path: Ruta al archivo de audio candidato proporcionado por el usuario.
        """
        # Verificar si el archivo existe
        if not os.path.isfile(audio_candidato_path):
            raise FileNotFoundError(f"El archivo {audio_candidato_path} no existe o no es accesible.")

        # Procesar el audio
        processed_audio = self._process_audio(audio_candidato_path)

        # Guardar el audio procesado en la carpeta `Candidato`
        output_path = os.path.join(self.candidato_path, "candidato_procesado.wav")
        sf.write(output_path, processed_audio, self.sampling_rate)  # Guardar como archivo WAV
        print(f"Audio candidato procesado y guardado en: {output_path}")