import os
import json
import traceback
from datetime import datetime
import torch
import soundfile as sf
import librosa
from pathlib import Path
from qwen_tts import Qwen3TTSModel

def registrar_log(mensagem: str):
    linha = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {mensagem}"
    print(linha, flush=True)
    try:
        with open("sistema.log", "a", encoding="utf-8") as f:
            f.write(linha + "\n")
    except Exception:
        pass

class TTSService:
    def __init__(self, model_path: str, dict_file: str = "phoneme_dict.json"):
        registrar_log("=== Inicializando o Sintetizador Qwen3-TTS (Modo Local / CPU) ===")
        self.device = "cpu"
        self.dict_file = dict_file
        
        caminho_modelo_local = Path(model_path)
        checkpoint = caminho_modelo_local / "model.safetensors"
        config = caminho_modelo_local / "config.json"

        if not checkpoint.is_file() or not config.is_file():
            registrar_log(f"[ERRO]: Os arquivos do modelo nao foram encontrados em {caminho_modelo_local}")
        else:
            registrar_log("Arquivos verificados com sucesso na pasta local.")

        registrar_log("Carregando modelo na CPU na Main Thread (isso pode levar alguns segundos)...")
        self.model = Qwen3TTSModel.from_pretrained(
            str(caminho_modelo_local),
            device_map=self.device,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=False
        )
        registrar_log("Modelo carregado com sucesso!")
    def _carregar_dicionario(self) -> dict:
        if os.path.exists(self.dict_file):
            try:
                with open(self.dict_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                registrar_log(f"[AVISO] Erro ao ler {self.dict_file}: {e}")
        return {}

    def aplicar_dicionario_fonetico(self, texto: str) -> str:
        dicionario = self._carregar_dicionario()
        texto_modificado = texto
        for termo_original, fonetica in dicionario.items():
            texto_modificado = texto_modificado.replace(f" {termo_original} ", f" {fonetica} ")
            if texto_modificado.startswith(f"{termo_original} "):
                texto_modificado = texto_modificado.replace(f"{termo_original} ", f"{fonetica} ", 1)
        if texto_modificado != texto:
            registrar_log(f"Fonetica aplicada: '{texto}' -> '{texto_modificado}'")
        return texto_modificado

    def synthesize(self, ref_audio_path: str, ref_text: str, text_to_speak: str, output_filename: str = "resultado_voz.wav") -> str:
        if not ref_audio_path or not os.path.exists(ref_audio_path):
            raise ValueError(f"Audio de referencia nao encontrado: {ref_audio_path}")

        try:
            if not ref_audio_path.endswith(".wav"):
                y, sr_orig = librosa.load(ref_audio_path, sr=16000)
                audio_processado = "referencia_temp.wav"
                sf.write(audio_processado, y, sr_orig)
            else:
                audio_processado = ref_audio_path

            texto_final = self.aplicar_dicionario_fonetico(text_to_speak)
            registrar_log(f"Iniciando geracao de voz para ({len(texto_final)} caracteres): '{texto_final[:40]}...'")

            # Libera qualquer residuo de memoria antes da inferencia
            gc.collect()

            # Desativa rastreamento de gradientes para economizar RAM
            with torch.inference_mode():
                wavs, sr = self.model.generate_voice_clone(
                    text=texto_final,
                    language="Portuguese",
                    ref_audio=audio_processado,
                    ref_text=ref_text,
                )

            sf.write(output_filename, wavs[0], sr)
            registrar_log(f"Audio gerado com sucesso: {output_filename}")
            
            # Limpeza pos-geracao
            del wavs
            gc.collect()

            return output_filename
        except Exception as e:
            registrar_log(f"[ERRO NA GERACAO]:\n{traceback.format_exc()}")
            raise e