import sys
import os
import logging
import traceback

# Configura o Logging para gravar num arquivo físico além da consola
log_filename = "app_error.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Launcher")

logger.info("=== INICIANDO CAPTURA DE ARRANQUE ===")

try:
    logger.info("A importar app.py...")
    import app
    logger.info("Aplicação iniciada com sucesso.")
except Exception as e:
    logger.critical(f"EXCEÇÃO CRÍTICA CAPTURADA: {e}", exc_info=True)
    with open("crash_report.txt", "w", encoding="utf-8") as f:
        f.write(traceback.format_exc())
except BaseException as be:
    logger.critical(f"FALHA DE SISTEMA / SEGFAULT CAPTURADA: {be}", exc_info=True)

if __name__ == "__main__":
    pass