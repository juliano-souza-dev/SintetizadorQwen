@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ===================================================
echo   Inicializador - Sintetizador Qwen3
echo ===================================================

:: [1/3] Matar processos python travados na memoria RAM e limpar cache
echo [1/3] Limpando memoria de processos anteriores e cache...
taskkill /F /IM python.exe /T >nul 2>&1
if exist "__pycache__" rd /s /q "__pycache__"

:: [2/3] Verificar Python
if not exist "python\python.exe" (
    echo [ERRO] python\python.exe nao encontrado!
    pause
    exit /b 1
)

:: [3/3] Iniciar aplicacao mostrando na tela em tempo real
echo [2/3] Iniciando a aplicacao...
echo.
python\python.exe -u -X faulthandler app.py

if errorlevel 1 (
    echo.
    echo ===================================================
    echo [ALERTA] O processo caiu com codigo: %errorlevel%
    echo Verifique o arquivo "sistema.log" para ver o historico.
    echo ===================================================
)
pause