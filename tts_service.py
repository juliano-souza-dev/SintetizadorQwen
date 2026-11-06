from __future__ import annotations

import gc
import json
import logging
import os
import re
import traceback
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

UTC = timezone.utc

import librosa
import numpy as np
from qwen_tts import Qwen3TTSModel
import soundfile as sf
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_XET"] = "1"


def registrar_log(mensagem: str) -> None:
    # Usa logging padrão para que os logs de síntese também entrem no arquivo
    # individual do job enquanto ele estiver em execução.
    logging.getLogger("tts_service").info(mensagem)

class TTSService:
    def __init__(self, model_path: str, dict_file: str = "phoneme_dict.json") -> None:
        registrar_log("=== Inicializando TTSService (Qwen3) ===")
        self.device = "cpu"
        self.dict_file = dict_file

        caminho_modelo_local = Path(model_path)
        checkpoint_file = caminho_modelo_local / "model.safetensors"
        config = caminho_modelo_local / "config.json"

        if not caminho_modelo_local.is_dir():
            raise FileNotFoundError(
                f"Diretório do modelo Qwen3-TTS não encontrado: {caminho_modelo_local}"
            )
        if not checkpoint_file.is_file() or not config.is_file():
            faltantes = [
                str(p) for p in (checkpoint_file, config) if not p.is_file()
            ]
            raise FileNotFoundError(
                "Arquivos obrigatórios do modelo não encontrados: " + ", ".join(faltantes)
            )

        registrar_log("Arquivos do modelo verificados.")
        gc.collect()
        registrar_log(
            "Carregando modelo na CPU (dtype=bfloat16, low_cpu_mem_usage=False)..."
        )
        try:
            self.model = Qwen3TTSModel.from_pretrained(
                str(caminho_modelo_local),
                device_map=self.device,
                dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
            )
        except Exception:
            registrar_log("[FATAL] Falha ao carregar Qwen3-TTS:")
            registrar_log(traceback.format_exc())
            raise
        registrar_log("Modelo carregado com sucesso no worker!")

    def _carregar_dicionario(self) -> dict:
        if os.path.exists(self.dict_file):
            try:
                with open(self.dict_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                registrar_log(f"[AVISO] Erro ao ler {self.dict_file}: {e}")
        return {}

    def limpar_e_otimizar_texto(self, texto: str) -> str:
        texto_limpo = texto.replace('"', "").replace("'", "")
        texto_limpo = re.sub(
            r"\bIm\b", "I am", texto_limpo, flags=re.IGNORECASE
        )
        texto_limpo = re.sub(
            r"\bI'm\b", "I am", texto_limpo, flags=re.IGNORECASE
        )
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

    def _preparar_audio_referencia(
        self, ref_audio_path: str, output_path: str
    ) -> tuple[str, bool]:
        if not ref_audio_path.lower().endswith(".wav"):
            y, sr_orig = librosa.load(ref_audio_path, sr=16000)
            unique_id = uuid.uuid4().hex[:8]
            temp_ref = Path(output_path).parent / f"temp_ref_{unique_id}.wav"
            sf.write(str(temp_ref), y, sr_orig)
            return str(temp_ref), True
        return ref_audio_path, False


    def _attach_audio_token_monitor(
        self,
        progress_callback: Callable[[dict], None] | None,
        *,
        scope: str = "direct",
        current_chunk: int | None = None,
        total_chunks: int | None = None,
    ):
        """Observa os códigos de áudio emitidos pelo Talker sem alterar a geração."""
        if progress_callback is None:
            return None

        model_core = getattr(self.model, "model", None)
        talker = getattr(model_core, "talker", None)
        if talker is None or not hasattr(talker, "register_forward_hook"):
            registrar_log("[AVISO] Talker não expõe hook; contador de tokens indisponível.")
            return None

        state = {
            "generated_audio_tokens": 0,
            "generated_audio_frames": 0,
            "started_at": time.monotonic(),
            "last_report_at": 0.0,
        }

        def on_talker_forward(_module, _inputs, output):
            try:
                hidden_states = getattr(output, "hidden_states", None)
                if not isinstance(hidden_states, (tuple, list)) or len(hidden_states) < 2:
                    return
                codec_ids = hidden_states[1]
                if not isinstance(codec_ids, torch.Tensor) or codec_ids.numel() == 0:
                    return

                # Cada linha representa um passo temporal; as colunas são os
                # codebooks produzidos para esse frame de áudio.
                state["generated_audio_tokens"] += int(codec_ids.numel())
                state["generated_audio_frames"] += (
                    int(codec_ids.shape[0]) if codec_ids.ndim >= 2 else 1
                )

                now = time.monotonic()
                if state["last_report_at"] and now - state["last_report_at"] < 0.8:
                    return
                state["last_report_at"] = now
                elapsed = max(now - state["started_at"], 0.001)
                payload = {
                    "phase": "synthesizing",
                    "event": "audio_token_progress",
                    "progress_scope": scope,
                    "generated_audio_tokens": state["generated_audio_tokens"],
                    "generated_audio_frames": state["generated_audio_frames"],
                    "tokens_per_second": round(state["generated_audio_tokens"] / elapsed, 2),
                    "inference_elapsed_seconds": round(elapsed, 2),
                }
                if current_chunk is not None:
                    payload["current_chunk"] = current_chunk
                if total_chunks is not None:
                    payload["total_chunks"] = total_chunks
                try:
                    progress_callback(payload)
                except Exception:
                    # A perda do canal de progresso não deve cancelar a síntese.
                    registrar_log("[AVISO] Não foi possível enviar atualização de tokens.")
            except Exception:
                # Instrumentação é observacional: nunca pode interromper o Qwen.
                registrar_log("[AVISO] Falha ao ler os tokens internos do Qwen.")

        handle = talker.register_forward_hook(on_talker_forward)
        registrar_log("Monitoramento de tokens de áudio conectado ao Talker.")
        return handle, state

    @staticmethod
    def _audio_token_metrics(monitor) -> dict:
        if monitor is None:
            return {}
        _handle, state = monitor
        elapsed = max(time.monotonic() - state["started_at"], 0.001)
        return {
            "generated_audio_tokens": int(state["generated_audio_tokens"]),
            "generated_audio_frames": int(state["generated_audio_frames"]),
            "tokens_per_second": round(
                state["generated_audio_tokens"] / elapsed, 2
            ),
            "inference_elapsed_seconds": round(elapsed, 2),
        }

    @staticmethod
    def _detach_audio_token_monitor(monitor) -> None:
        if monitor is not None:
            try:
                monitor[0].remove()
            except Exception:
                pass

    def synthesize(
        self,
        ref_audio_path: str,
        ref_text: str,
        text_to_speak: str,
        output_path: str,
        temperature: float = 0.65,
        checkpoint: Callable[[], None] | None = None,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> str:
        """Executa síntese direta e reporta o tempo de cada etapa."""

        def report(event: str, **details) -> None:
            data = {
                "phase": "synthesizing",
                "event": event,
                "text_chars": len(text_to_speak or ""),
                "timestamp": datetime.now(UTC).isoformat(),
                **details,
            }
            registrar_log(
                f"DIRECT_SYNTHESIS event={event} details={details}"
            )
            if progress_callback is not None:
                progress_callback(data)

        total_started = time.monotonic()
        if not ref_audio_path or not os.path.exists(ref_audio_path):
            raise ValueError(
                f"Áudio de referência não encontrado: {ref_audio_path}"
            )

        if checkpoint:
            checkpoint()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        preparation_started = time.monotonic()
        report("reference_preparation_started")
        audio_processado, is_temp = self._preparar_audio_referencia(
            ref_audio_path, output_path
        )
        preparation_seconds = time.monotonic() - preparation_started
        report(
            "reference_preparation_completed",
            duration_seconds=round(preparation_seconds, 3),
        )

        try:
            texto_otimizado = self.limpar_e_otimizar_texto(text_to_speak)
            texto_final = self.aplicar_dicionario_fonetico(texto_otimizado)

            if checkpoint:
                checkpoint()

            registrar_log(
                f"Iniciando síntese direta ({len(texto_final)} caracteres): "
                f"'{texto_final[:80]}'"
            )
            gc.collect()

            inference_started = time.monotonic()
            report(
                "inference_started",
                language="Portuguese",
                temperature=temperature,
                prepared_text_chars=len(texto_final),
            )
            token_monitor = self._attach_audio_token_monitor(
                progress_callback, scope="direct"
            )
            try:
                with torch.inference_mode():
                    wavs, sr = self.model.generate_voice_clone(
                        text=texto_final,
                        language="Portuguese",
                        ref_audio=audio_processado,
                        ref_text=ref_text,
                        temperature=temperature,
                    )
            finally:
                self._detach_audio_token_monitor(token_monitor)

            inference_seconds = time.monotonic() - inference_started
            token_metrics = self._audio_token_metrics(token_monitor)
            report(
                "inference_completed",
                duration_seconds=round(inference_seconds, 3),
                sample_rate=int(sr),
                **token_metrics,
            )

            write_started = time.monotonic()
            report("audio_write_started")
            sf.write(output_path, wavs[0], sr)
            write_seconds = time.monotonic() - write_started
            report(
                "audio_write_completed",
                duration_seconds=round(write_seconds, 3),
                output_path=output_path,
            )
            registrar_log(f"Áudio salvo em: {output_path}")
            del wavs

            total_seconds = time.monotonic() - total_started
            report(
                "direct_completed",
                preparation_seconds=round(preparation_seconds, 3),
                inference_seconds=round(inference_seconds, 3),
                audio_write_seconds=round(write_seconds, 3),
                total_seconds=round(total_seconds, 3),
            )
            return output_path
        finally:
            if is_temp and os.path.exists(audio_processado):
                try:
                    os.remove(audio_processado)
                except Exception:
                    pass
            gc.collect()

    def synthesize_chunks(
        self,
        ref_audio_path: str,
        ref_text: str,
        chunks: list[dict],
        output_path: str,
        temperature: float = 0.65,
        checkpoint: Callable[[], None] | None = None,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> str:
        if not ref_audio_path or not os.path.exists(ref_audio_path):
            raise ValueError(
                f"Áudio de referência não encontrado: {ref_audio_path}"
            )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        audio_processado, is_temp = self._preparar_audio_referencia(
            ref_audio_path, output_path
        )

        segmentos_audio = []
        taxa_amostragem = 16000
        chunk_history: list[dict] = []

        try:
            for idx, chunk in enumerate(chunks, 1):
                if checkpoint:
                    checkpoint()

                raw_text = chunk.get("text", "").strip()
                if not raw_text:
                    continue

                lang_code = chunk.get("lang", "pt").lower()
                idioma_qwen = (
                    "English"
                    if lang_code in ["en", "english", "inglês", "ingles"]
                    else "Portuguese"
                )
                pausa_ms = chunk.get("pause_after_ms", 150)
                chunk_started_at = datetime.now(UTC).isoformat()
                chunk_started_monotonic = time.monotonic()
                remaining_chars = sum(
                    len(str(item.get("text", "")).strip())
                    for item in chunks[idx - 1:]
                    if isinstance(item, dict)
                )
                if progress_callback:
                    progress_callback({
                        "phase": "synthesizing",
                        "event": "chunk_started",
                        "current_chunk": idx,
                        "total_chunks": len(chunks),
                        "completed_chunks": idx - 1,
                        "chunk_text": raw_text,
                        "chunk_chars": len(raw_text),
                        "remaining_chars": remaining_chars,
                        "chunk_started_at": chunk_started_at,
                        "chunk_history": list(chunk_history),
                    })

                texto_chunk = self.limpar_e_otimizar_texto(raw_text)
                texto_chunk = self.aplicar_dicionario_fonetico(texto_chunk)

                registrar_log(
                    f"[Chunk {idx}/{len(chunks)}] Gerando ({idioma_qwen}): '{texto_chunk}'"
                )
                gc.collect()

                token_monitor = self._attach_audio_token_monitor(
                    progress_callback,
                    scope="chunk",
                    current_chunk=idx,
                    total_chunks=len(chunks),
                )
                try:
                    with torch.inference_mode():
                        wavs, sr = self.model.generate_voice_clone(
                            text=texto_chunk,
                            language=idioma_qwen,
                            ref_audio=audio_processado,
                            ref_text=ref_text,
                            temperature=temperature,
                        )
                finally:
                    self._detach_audio_token_monitor(token_monitor)

                chunk_duration = time.monotonic() - chunk_started_monotonic
                token_metrics = self._audio_token_metrics(token_monitor)
                chunk_history.append({
                    "text_chars": len(texto_chunk),
                    "duration_seconds": round(chunk_duration, 3),
                    **token_metrics,
                })
                taxa_amostragem = sr
                audio_array = wavs[0]
                segmentos_audio.append(audio_array)

                if pausa_ms > 0:
                    amostras_silencio = int(
                        (pausa_ms / 1000.0) * taxa_amostragem
                    )
                    silencio = np.zeros(
                        amostras_silencio, dtype=audio_array.dtype
                    )
                    segmentos_audio.append(silencio)

                del wavs
                if progress_callback:
                    progress_callback({
                        "phase": "synthesizing",
                        "event": "chunk_completed",
                        "current_chunk": idx,
                        "total_chunks": len(chunks),
                        "completed_chunks": idx,
                        "chunk_text": raw_text,
                        "chunk_chars": len(texto_chunk),
                        "remaining_chars": sum(
                            len(str(item.get("text", "")).strip())
                            for item in chunks[idx:]
                            if isinstance(item, dict)
                        ),
                        "last_chunk_duration_seconds": round(chunk_duration, 3),
                        **token_metrics,
                        "chunk_history": list(chunk_history),
                    })

            if progress_callback:
                progress_callback({
                    "phase": "finalizing",
                    "event": "assembling_audio",
                    "current_chunk": len(chunks),
                    "total_chunks": len(chunks),
                    "completed_chunks": len(chunks),
                    "chunk_history": list(chunk_history),
                })

            if not segmentos_audio:
                raise ValueError("Nenhum áudio gerado a partir dos chunks.")

            audio_final = np.concatenate(segmentos_audio, axis=0)
            sf.write(output_path, audio_final, taxa_amostragem)
            registrar_log(
                f"Áudio concatenado salvo com sucesso em: {output_path}"
            )
            return output_path
        finally:
            if is_temp and os.path.exists(audio_processado):
                try:
                    os.remove(audio_processado)
                except Exception:
                    pass
            gc.collect()