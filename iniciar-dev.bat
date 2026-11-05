@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Gerenciador DEV - Sintetizador Qwen3-TTS

echo ===================================================
echo   Inicializador DEV - Qwen3-TTS (Worker + Auto Reload)
echo ===================================================

:: [1/5] Verificar executavel do Python portatil
echo [1/5] Verificando ambiente Python local...
if not exist "python\python.exe" (
    echo [ERRO CRITICO] python\python.exe nao foi encontrado na pasta do projeto!
    pause
    exit /b 1
)

:: [2/5] Checagem e instalacao automatica das mesmas dependencias do iniciar.bat
echo [2/5] Verificando dependencias essenciais...
set "FALTANDO="
if not exist "python\Lib\site-packages\torch" set "FALTANDO=1"
if not exist "python\Lib\site-packages\gradio" set "FALTANDO=1"
if not exist "python\Lib\site-packages\qwen_tts" set "FALTANDO=1"
if not exist "python\Lib\site-packages\soundfile.py" if not exist "python\Lib\site-packages\soundfile" set "FALTANDO=1"
if not exist "python\Lib\site-packages\librosa" set "FALTANDO=1"

if defined FALTANDO (
    echo [AVISO] Pacotes ausentes detectados. Instalando dependencias necessarias...
    python\python.exe -m pip install --disable-pip-version-check --no-input torch gradio qwen-tts soundfile librosa numpy huggingface_hub
    if errorlevel 1 (
        echo [ERRO CRITICO] Falha ao instalar as dependencias. Verifique a conexao com a internet.
        pause
        exit /b 1
    )
    echo       Dependencias instaladas com sucesso!
) else (
    echo       Todas as dependencias estao prontas.
)

:: [3/5] Criar as mesmas pastas de dados do iniciar.bat
echo [3/5] Validando diretorios estruturais...
if not exist "data" mkdir "data"
if not exist "data\outputs" mkdir "data\outputs"
if not exist "data\voices" mkdir "data\voices"
if not exist "logs" mkdir "logs"

:: [4/5] Iniciar o mesmo Worker TTS em janela dedicada
echo [4/5] Subindo o Worker TTS em processo isolado...
set HF_HUB_DISABLE_XET=1
set KMP_DUPLICATE_LIB_OK=TRUE

start "Qwen3-TTS Worker (Backend)" cmd /k "chcp 65001 >nul && cd /d "%~dp0" && echo === WORKER TTS ATIVO === && python\python.exe -u worker_tts.py"

:: Pequena pausa para permitir que o worker inicialize o banco SQLite
timeout /t 3 /nobreak >nul

:: [5/5] Iniciar supervisor DEV do Gradio
echo [5/5] Subindo servidor DEV com reinicio automatico...
echo.
echo Salve qualquer arquivo .py para reiniciar o servidor.
echo A aba do navegador sera aberta uma unica vez e nao sera recarregada pelo supervisor.
echo Pressione Ctrl+C para encerrar.
echo.

python\python.exe -u dev_server.py

echo.
echo Encerrando processos vinculados...
taskkill /FI "WINDOWTITLE eq Qwen3-TTS Worker (Backend)*" /T /F >nul 2>&1
echo Concluido.
pause
