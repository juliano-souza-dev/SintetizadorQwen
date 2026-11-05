import os
import gc
import json
import re
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np
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

    def limpar_e_otimizar_texto(self, texto: str) -> str:
        """
        Aplica as regras de pontuação rítmica:
        - Remove aspas para evitar hesitação/leitura literal do tokenizer.
        - Expande contrações críticas como 'I'm' -> 'I am' para firmar a chave fonética.
        """
        texto_limpo = texto.replace('"', '').replace("'", "")
        # Expande I'm para I am se isolado
        texto_limpo = re.sub(r"\bIm\b", "I am", texto_limpo, flags=re.IGNORECASE)
        texto_limpo = re.sub(r"\bI'm\b", "I am", texto_limpo, flags=re.IGNORECASE)
        return texto_limpo

    def aplicar_dicionario_fonetico(self, texto: str) -> str:
        dicionario = self._carregar_dicionario()
        texto_modificado = texto
        for termo_original, fonetica in dicionario.items():
            padrao = re.compile(rf"(?<!\w){re.escape(termo_original)}(?!\w)")
            texto_modificado = padrao.sub(fonetica, texto_modificado)
        if texto_modificado != texto:
            registrar_log(f"Fonética aplicada: '{texto}' -> '{texto_modificado}'")
        return texto_modificado

    def preparar_audio_referencia(self, ref_audio_path: str, output_path: str) -> str:
        if not ref_audio_path.lower().endswith(".wav"):
            y, sr_orig = librosa.load(ref_audio_path, sr=16000)
            temp_ref = Path(output_path).parent / "temp_ref.wav"
            sf.write(str(temp_ref), y, sr_orig)
            return str(temp_ref)
        return ref_audio_path

    def synthesize(self, ref_audio_path: str, ref_text: str, text_to_speak: str, output_path: str, temperature: float = 0.65) -> str:
        """Síntese direta em prompt único com pré-processamento otimizado."""
        if not ref_audio_path or not os.path.exists(ref_audio_path):
            raise ValueError(f"Áudio de referência não encontrado: {ref_audio_path}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            audio_processado = self.preparar_audio_referencia(ref_audio_path, output_path)
            texto_otimizado = self.limpar_e_otimizar_texto(text_to_speak)
            texto_final = self.aplicar_dicionario_fonetico(texto_otimizado)
            
            registrar_log(f"Iniciando síntese direta: '{texto_final[:45]}...'")

            gc.collect()
            with torch.inference_mode():
                wavs, sr = self.model.generate_voice_clone(
                    text=texto_final,
                    language="Portuguese",
                    ref_audio=audio_processado,
                    ref_text=ref_text,
                    temperature=temperature
                )

            sf.write(output_path, wavs[0], sr)
            registrar_log(f"Áudio salvo em: {output_path}")

            del wavs
            if audio_processado.endswith("temp_ref.wav") and os.path.exists(audio_processado):
                try:
                    os.remove(audio_processado)
                except Exception:
                    pass
            gc.collect()

            return output_path
        except Exception as e:
            registrar_log(f"[ERRO NA SÍNTESE]:\n{traceback.format_exc()}")
            raise e

    def synthesize_chunks(self, ref_audio_path: str, ref_text: str, chunks: list, output_path: str, temperature: float = 0.65) -> str:
        """
        Síntese por chunking com troca nativa de idioma por bloco 
        e concatenação de áudio com micro-silêncios via NumPy.
        """
        if not ref_audio_path or not os.path.exists(ref_audio_path):
            raise ValueError(f"Áudio de referência não encontrado: {ref_audio_path}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        audio_processado = self.preparar_audio_referencia(ref_audio_path, output_path)

        segmentos_audio = []
        taxa_amostragem = 16000

        try:
            for idx, chunk in enumerate(chunks, 1):
                raw_text = chunk.get("text", "").strip()
                if not raw_text:
                    continue

                lang_code = chunk.get("lang", "pt").lower()
                idioma_qwen = "English" if lang_code in ["en", "english", "inglês", "ingles"] else "Portuguese"
                pausa_ms = chunk.get("pause_after_ms", 150)

                texto_chunk = self.limpar_e_otimizar_texto(raw_text)
                texto_chunk = self.aplicar_dicionario_fonetico(texto_chunk)

                registrar_log(f"[Chunk {idx}/{len(chunks)}] Gerando ({idioma_qwen}): '{texto_chunk}'")

                gc.collect()
                with torch.inference_mode():
                    wavs, sr = self.model.generate_voice_clone(
                        text=texto_chunk,
                        language=idioma_qwen,
                        ref_audio=audio_processado,
                        ref_text=ref_text,
                        temperature=temperature
                    )
                
                taxa_amostragem = sr
                audio_array = wavs[0]
                segmentos_audio.append(audio_array)

                # Cria o silêncio preciso em milissegundos
                if pausa_ms > 0:
                    amostras_silencio = int((pausa_ms / 1000.0) * taxa_amostragem)
                    silencio = np.zeros(amostras_silencio, dtype=audio_array.dtype)
                    segmentos_audio.append(silencio)

                del wavs

            if not segmentos_audio:
                raise ValueError("Nenhum áudio gerado a partir dos chunks.")

            # Concatena os blocos num único áudio contínuo
            audio_final = np.concatenate(segmentos_audio, axis=0)
            sf.write(output_path, audio_final, taxa_amostragem)
            registrar_log(f"Áudio com chunks concatenados gravado com sucesso em: {output_path}")

            if audio_processado.endswith("temp_ref.wav") and os.path.exists(audio_processado):
                try:
                    os.remove(audio_processado)
                except Exception:
                    pass

            gc.collect()
            return output_path

        except Exception as e:
            registrar_log(f"[ERRO NO CHUNKING]:\n{traceback.format_exc()}")
            raise e