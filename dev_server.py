from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

BASE_DIR = Path(__file__).resolve().parent
APP_FILE = BASE_DIR / "app.py"
APP_URL = "http://127.0.0.1:7860/"
IGNORED_DIRS = {
    ".git", "__pycache__", "python", "venv", ".venv",
    "qwen-venv", "data", "logs", "outputs",
}


def source_snapshot() -> dict[str, tuple[int, int]]:
    """Retorna mtime e tamanho dos fontes, ignorando ambientes e dados."""
    snapshot: dict[str, tuple[int, int]] = {}
    for root, dirs, files in os.walk(BASE_DIR):
        dirs[:] = [
            name for name in dirs
            if name not in IGNORED_DIRS and not name.startswith(".")
        ]
        for filename in files:
            if not filename.endswith(".py"):
                continue
            path = Path(root) / filename
            try:
                stat = path.stat()
                snapshot[str(path.relative_to(BASE_DIR))] = (
                    stat.st_mtime_ns, stat.st_size
                )
            except OSError:
                continue
    return snapshot


def start_app() -> subprocess.Popen:
    env = os.environ.copy()
    env["QWEN_DEV_MODE"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    print("\n[DEV] Iniciando servidor Gradio...", flush=True)
    return subprocess.Popen(
        [sys.executable, "-u", str(APP_FILE)],
        cwd=BASE_DIR,
        env=env,
    )


def stop_app(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    print("[DEV] Encerrando servidor anterior...", flush=True)
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def wait_until_ready(process: subprocess.Popen, timeout: float = 180.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urlopen(APP_URL, timeout=1) as response:
                if response.status < 500:
                    return True
        except (URLError, TimeoutError, OSError):
            pass
        time.sleep(0.5)
    return False


def main() -> int:
    if not APP_FILE.is_file():
        print(f"[ERRO] app.py não encontrado: {APP_FILE}")
        return 1

    print("=" * 62)
    print(" QWEN3-TTS | MODO DESENVOLVIMENTO")
    print(" Auto-restart do servidor ao detectar alterações em arquivos .py")
    print(" A aba do navegador será aberta uma única vez.")
    print("=" * 62)

    snapshot = source_snapshot()
    process = start_app()
    browser_opened = False

    try:
        while True:
            if not browser_opened:
                if wait_until_ready(process):
                    webbrowser.open(APP_URL)
                    browser_opened = True
                    print(f"[DEV] Aplicação disponível em {APP_URL}", flush=True)
                elif process.poll() is not None:
                    print(
                        "[DEV] O servidor encerrou durante a inicialização. "
                        "Corrija o erro exibido e salve um arquivo .py para tentar novamente.",
                        flush=True,
                    )
                    while process.poll() is not None:
                        time.sleep(0.5)
                        current = source_snapshot()
                        if current != snapshot:
                            snapshot = current
                            process = start_app()
                            break
                    continue

            time.sleep(0.8)
            current = source_snapshot()
            if current != snapshot:
                added = sorted(set(current) - set(snapshot))
                removed = sorted(set(snapshot) - set(current))
                changed = sorted(
                    name for name in set(current) & set(snapshot)
                    if current[name] != snapshot[name]
                )
                print("\n[DEV] Alteração detectada:", flush=True)
                for name in added:
                    print(f"  [+] {name}", flush=True)
                for name in changed:
                    print(f"  [~] {name}", flush=True)
                for name in removed:
                    print(f"  [-] {name}", flush=True)

                snapshot = current
                # Pequeno debounce para agrupar salvamentos consecutivos do editor.
                time.sleep(0.7)
                snapshot = source_snapshot()
                stop_app(process)
                process = start_app()
                if wait_until_ready(process):
                    print(
                        "[DEV] Servidor reiniciado. A aba existente foi mantida.",
                        flush=True,
                    )
                else:
                    print(
                        "[DEV] O servidor não voltou. Consulte o erro acima; "
                        "ele tentará iniciar novamente após uma nova alteração.",
                        flush=True,
                    )

            elif process.poll() is not None:
                print(
                    f"[DEV] Servidor encerrou (código {process.returncode}). "
                    "Aguardando alteração no código.",
                    flush=True,
                )
                while process.poll() is not None:
                    time.sleep(0.5)
                    current = source_snapshot()
                    if current != snapshot:
                        snapshot = current
                        process = start_app()
                        break

    except KeyboardInterrupt:
        print("\n[DEV] Encerrando modo desenvolvimento...", flush=True)
        stop_app(process)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
