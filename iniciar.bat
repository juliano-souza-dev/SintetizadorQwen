@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Gerenciador - Sintetizador Qwen3-TTS

echo ===================================================
echo   Inicializador Unificado - Qwen3-TTS (Back + Front)
echo ===================================================

:: [1/5] Verificar executavel do Python portatil
echo [1/5] Verificando ambiente Python local...
if not exist "python\python.exe" (
    echo [ERRO CRITICO] python\python.exe nao foi encontrado na pasta do projeto!
    pause
    exit /b 1
)

:: [2/5] Checagem e instalacao automatica de dependencias
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

:: [3/5] Criar pastas de dados caso nao existam
echo [3/5] Validando diretorios estruturais...
if not exist "data" mkdir "data"
if not exist "data\outputs" mkdir "data\outputs"
if not exist "data\voices" mkdir "data\voices"

:: [4/5] Iniciar supervisor do Worker TTS em janela dedicada.
:: Se o worker cair por erro Python, crash nativo ou encerramento inesperado,
:: o supervisor inicia uma nova instancia automaticamente.
echo [4/5] Subindo o supervisor do Worker TTS...
set HF_HUB_DISABLE_XET=1
set KMP_DUPLICATE_LIB_OK=TRUE

start "Qwen3-TTS Worker Supervisor" cmd /k "chcp 65001 >nul && cd /d "%~dp0" && echo === SUPERVISOR TTS ATIVO === && python\python.exe -u worker_supervisor.py"

:: Pequena pausa para o worker iniciar o banco SQLite antes do front
timeout /t 3 /nobreak >nul

:: [5/5] Iniciar a Interface Gradio (Frontend)
echo [5/5] Subindo a interface Gradio em http://127.0.0.1:7860 ...
echo Pressione Ctrl+C para encerrar a interface.
echo.

python\python.exe -u app.py

:: Ao fechar a interface, encerra o supervisor e o worker vinculado.
echo.
echo Encerrando processos vinculados...
taskkill /FI "WINDOWTITLE eq Qwen3-TTS Worker Supervisor*" /T /F >nul 2>&1
echo Concluido.
pause
