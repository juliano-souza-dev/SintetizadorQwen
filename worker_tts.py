from __future__ import annotations

import logging
import faulthandler
import threading
import warnings
import os
from pathlib import Path
import signal
import socket
import sys
from datetime import timedelta
from threading import Event

# 1. Garante que o diretório atual do projeto está no início do sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# 2. Importações de infraestrutura e serviços locais
from persistent_worker import PersistentWorker
from sqlite_job_repository import SqliteJobRepository
from tts_job_handler import make_tts_handler
from inference_process import InferenceProcessClient
import atexit

# 3. Definição do caminho do modelo de IA
CAMINHO_MODELO = os.environ.get(
    "QWEN3_MODEL_PATH",
    r"C:\Users\jsouz\Projects\nova-generator\models\qwen3-tts\Qwen3-TTS-12Hz-1.7B-Base",
)

# 4. Configuração isolada de registo de eventos (logs) para o processo worker
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "worker_tts.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("worker_tts")

# Captura warnings Python e falhas fatais nativas (ex.: crash em extensão C/CUDA).
warnings.simplefilter("default")
logging.captureWarnings(True)
_FATAL_LOG_HANDLE = open(BASE_DIR / "worker_fatal.log", "a", encoding="utf-8")
faulthandler.enable(file=_FATAL_LOG_HANDLE, all_threads=True)


def _log_uncaught_exception(exc_type, exc_value, exc_traceback) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical(
        "EXCEÇÃO NÃO CAPTURADA NO PROCESSO WORKER",
        exc_info=(exc_type, exc_value, exc_traceback),
    )


sys.excepthook = _log_uncaught_exception


def _log_thread_exception(args) -> None:
    logger.critical(
        "EXCEÇÃO NÃO CAPTURADA NA THREAD %s",
        args.thread.name if args.thread else "desconhecida",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


threading.excepthook = _log_thread_exception


def main() -> None:
    logger.info("=== A iniciar Worker TTS Dedicado ===")

    # Instancia o repositório durável partilhado em SQLite
    repository = SqliteJobRepository(db_path=str(BASE_DIR / "data" / "jobs.db"))

    # A inferência Qwen roda em processo separado. O processo mantém o
    # modelo carregado entre jobs e pode ser encerrado sem matar a fila.
    inference_client = InferenceProcessClient(
        model_path=CAMINHO_MODELO,
        dict_file=str(BASE_DIR / "phoneme_dict.json"),
        timeout_seconds=float(os.environ.get("QWEN_INFERENCE_TIMEOUT_SECONDS", "900")),
        log_path=str(BASE_DIR / "worker_inference.log"),
    )
    atexit.register(inference_client.close)

    def get_engine():
        return inference_client

    # Identificador exclusivo deste processo de execução
    worker_id = f"worker-tts-{socket.gethostname()}-{os.getpid()}"
    logger.info(f"Worker registado com ID: {worker_id}")

    # Regista o handler específico para tarefas de síntese de voz
    handlers = {"tts_generate": make_tts_handler(get_engine)}

    worker = PersistentWorker(
        repository=repository,
        worker_id=worker_id,
        handlers=handlers,
        heartbeat_timeout=timedelta(minutes=2),
        allowed_kinds=("tts_generate",),
        logger=logger,
    )

    stop_event = Event()

    def handle_exit(signum: int, frame: object) -> None:
        logger.info("Sinal de interrupção recebido. A encerrar worker de forma controlada...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    logger.info("Worker operacional e a monitorizar a fila de tarefas...")
    worker.run_forever(poll_interval_seconds=1.0, stop_event=stop_event)


if __name__ == "__main__":
    main()