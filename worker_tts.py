from __future__ import annotations

import logging
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
from tts_service import TTSService

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


def main() -> None:
    logger.info("=== A iniciar Worker TTS Dedicado ===")

    # Instancia o repositório durável partilhado em SQLite
    repository = SqliteJobRepository(db_path=str(BASE_DIR / "data" / "jobs.db"))

    # Carrega os pesos do Qwen3 na CPU (ocorre uma única vez na inicialização)
    logger.info(f"A carregar o motor TTS a partir de: {CAMINHO_MODELO}")
    try:
        engine = TTSService(
            model_path=CAMINHO_MODELO,
            dict_file=str(BASE_DIR / "phoneme_dict.json"),
        )
    except Exception:
        logger.exception(
            "FALHA FATAL NA INICIALIZAÇÃO DO QWEN3-TTS. "
            "O worker não irá consumir jobs."
        )
        raise

    # Identificador exclusivo deste processo de execução
    worker_id = f"worker-tts-{socket.gethostname()}-{os.getpid()}"
    logger.info(f"Worker registado com ID: {worker_id}")

    # Regista o handler específico para tarefas de síntese de voz
    handlers = {"tts_generate": make_tts_handler(engine)}

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