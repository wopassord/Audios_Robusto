from pydub import AudioSegment
import os

def convertir_ogg_a_wav(input_folder, output_folder):
    """
    Convierte todos los archivos .ogg en la carpeta de entrada a formato .wav
    utilizando ffmpeg como backend.
    
    Args:
        input_folder (str): Carpeta donde se encuentran los archivos .ogg.
        output_folder (str): Carpeta donde se guardarán los archivos .wav.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".ogg"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace(".ogg", ".wav"))

            try:
                # Cargar archivo OGG y exportar como WAV
                audio = AudioSegment.from_file(input_path, format="ogg")
                audio.export(output_path, format="wav")
                print(f"Convertido: {file_name} -> {output_path}")
            except Exception as e:
                print(f"Error al convertir {file_name}: {e}")

# Carpetas de entrada y salida
input_folder = r"C:\Users\berni\Desktop\OOG"
output_folder = r"C:\Users\berni\Desktop\WAV"

convertir_ogg_a_wav(input_folder, output_folder)
