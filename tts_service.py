import os
import gc
import json
import re
import traceback
from datetime import datetime
from pathlib import Path
import torch
import soundfile as sf
import librosa
from qwen_tts import Qwen3TTSModel

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_XET"] = "1"

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
        registrar_log("=== Inicializando TTSService (Qwen3) ===")
        self.device = "cpu"
        self.dict_file = dict_file
        
        caminho_modelo_local = Path(model_path)
        checkpoint = caminho_modelo_local / "model.safetensors"
        config = caminho_modelo_local / "config.json"

        if not checkpoint.is_file() or not config.is_file():
            registrar_log(f"[ERRO]: Arquivos do modelo não encontrados em {caminho_modelo_local}")
        else:
            registrar_log("Arquivos do modelo verificados.")

        gc.collect()
        registrar_log("Carregando modelo na CPU com bfloat16...")
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
            padrao = re.compile(rf"(?<!\w){re.escape(termo_original)}(?!\w)")
            texto_modificado = padrao.sub(fonetica, texto_modificado)
        if texto_modificado != texto:
            registrar_log(f"Fonética aplicada: '{texto}' -> '{texto_modificado}'")
        return texto_modificado

    def synthesize(self, ref_audio_path: str, ref_text: str, text_to_speak: str, output_path: str) -> str:
        if not ref_audio_path or not os.path.exists(ref_audio_path):
            raise ValueError(f"Áudio de referência não encontrado: {ref_audio_path}")

        # Garante que o diretório de destino exista
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            # Converte referência se não for WAV 16kHz
            if not ref_audio_path.lower().endswith(".wav"):
                y, sr_orig = librosa.load(ref_audio_path, sr=16000)
                temp_ref = Path(output_path).parent / "temp_ref.wav"
                sf.write(str(temp_ref), y, sr_orig)
                audio_processado = str(temp_ref)
            else:
                audio_processado = ref_audio_path

            texto_final = self.aplicar_dicionario_fonetico(text_to_speak)
            registrar_log(f"Iniciando síntese ({len(texto_final)} caracteres): '{texto_final[:40]}...'")

            gc.collect()
            with torch.inference_mode():
                wavs, sr = self.model.generate_voice_clone(
                    text=texto_final,
                    language="Portuguese",
                    ref_audio=audio_processado,
                    ref_text=ref_text,
                )

            sf.write(output_path, wavs[0], sr)
            registrar_log(f"Áudio salvo em: {output_path}")

            # Limpeza
            del wavs
            if not ref_audio_path.lower().endswith(".wav") and os.path.exists(audio_processado):
                try:
                    os.remove(audio_processado)
                except Exception:
                    pass
            gc.collect()

            return output_path
        except Exception as e:
            registrar_log(f"[ERRO NA SÍNTESE]:\n{traceback.format_exc()}")
            raise e