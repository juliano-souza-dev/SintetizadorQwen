from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = BASE_DIR / "worker_tts.py"
LOG_PATH = BASE_DIR / "worker_supervisor.log"

RESTART_INITIAL_SECONDS = 3
RESTART_MAX_SECONDS = 30
SHUTDOWN_TIMEOUT_SECONDS = 15


def configure_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return env


def main() -> int:
    if not WORKER_SCRIPT.is_file():
        print(f"[SUPERVISOR] Worker não encontrado: {WORKER_SCRIPT}", flush=True)
        return 2

    logger = logging.getLogger("worker_supervisor")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)

    stopping = False
    child: subprocess.Popen[bytes] | None = None

    def stop_handler(signum: int, frame: object) -> None:
        nonlocal stopping
        stopping = True
        logger.info("Sinal %s recebido; encerrando supervisor.", signum)
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGINT, stop_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop_handler)

    delay = RESTART_INITIAL_SECONDS
    logger.info("Supervisor iniciado. Script=%s Python=%s", WORKER_SCRIPT, sys.executable)

    try:
        while not stopping:
            started_at = time.monotonic()
            logger.info("Iniciando processo worker TTS.")
            try:
                with LOG_PATH.open("a", encoding="utf-8") as worker_log:
                    child = subprocess.Popen(
                        [sys.executable, "-u", str(WORKER_SCRIPT)],
                        cwd=str(BASE_DIR),
                        env=configure_environment(),
                        stdout=worker_log,
                        stderr=subprocess.STDOUT,
                    )
                    return_code = child.wait()
                child = None
            except OSError:
                logger.exception("Não foi possível iniciar o worker.")
                return_code = -1

            if stopping:
                break

            runtime = time.monotonic() - started_at
            logger.error(
                "Worker encerrou inesperadamente (exit_code=%s, runtime=%.1fs).",
                return_code,
                runtime,
            )

            # Se o worker ficou ativo por tempo razoável, reinicia com atraso curto.
            # Se está caindo em sequência, aplica backoff para evitar loop agressivo.
            if runtime >= 60:
                delay = RESTART_INITIAL_SECONDS
            logger.warning("Nova tentativa em %ss.", delay)
            end_at = time.monotonic() + delay
            while not stopping and time.monotonic() < end_at:
                time.sleep(min(0.5, end_at - time.monotonic()))
            delay = min(delay * 2, RESTART_MAX_SECONDS)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                logger.warning("Worker não encerrou no prazo; forçando encerramento.")
                child.kill()
                child.wait()
        logger.info("Supervisor encerrado.")
        handler.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
