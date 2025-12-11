@echo off

echo Setting up VS Code workspace...
if not exist ".vscode" mkdir .vscode

echo Creating .vscode\settings.json...
(
echo {
echo     "python.defaultInterpreterPath": "${workspaceFolder}/backend/.venv/Scripts/python.exe",
echo     "python.terminal.activateEnvironment": true,
echo     "python.terminal.activateEnvInCurrentTerminal": true,
echo     "python.analysis.extraPaths": [
echo         "${workspaceFolder}/backend"
echo     ],
echo     "python.envFile": "${workspaceFolder}/.env",
echo     "python.terminal.executeInFileDir": false,
echo     "python.pythonPath": "${workspaceFolder}/backend/.venv/Scripts/python.exe",
echo     "code-runner.executorMap": {
echo         "python": "\"$pythonPath\" -u $fullFileName"
echo     },
echo     "code-runner.fileDirectoryAsCwd": false,
echo     "code-runner.respectShebang": false,
echo     "python-envs.pythonProjects": [],
echo     "terminal.integrated.env.windows": {},
echo     "terminal.integrated.cwd": "${workspaceFolder}",
echo     "terminal.integrated.defaultProfile.windows": "Command Prompt",
echo     "terminal.integrated.profiles.windows": {
echo         "Command Prompt": {
echo             "path": "cmd.exe",
echo             "args": ["/K", "backend/.venv\\Scripts\\activate.bat"],
echo             "icon": "terminal-cmd"
echo         },
echo         "PowerShell": {
echo             "source": "PowerShell",
echo             "args": ["-NoExit", "-Command", "& 'backend/.venv\\Scripts\\Activate.ps1'"],
echo             "icon": "terminal-powershell"
echo         }
echo     },
echo     "terminal.integrated.automationProfile.windows": {
echo         "path": "cmd.exe",
echo         "args": ["/K", "backend/.venv\\Scripts\\activate.bat"]
echo     }
echo }
) > .vscode\settings.json


echo ========================================
echo LangGraph Backend - Clean Reinstall with UV
echo ========================================
echo.

echo [1/7] Setting working directory...
cd /d "%~dp0"
if %errorlevel% neq 0 (
    echo ERROR: Failed to change to script directory.
    pause
    exit /b 1
)
echo Working directory set to: %CD%
echo.

echo [2/7] Checking for UV installation...
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: UV is not installed. Please install UV first:
    echo curl -LsSf https://astral.sh/uv/install.sh | sh
    echo Or visit: https://github.com/astral-sh/uv
    pause
    exit /b 1
)
echo UV is installed.
echo.

echo [3/7] Removing existing virtual environment...
if exist ".venv" (
    echo Removing .venv directory...
    rmdir /s /q ".venv" 2>nul
    echo Virtual environment removed.
) else (
    echo No existing virtual environment found.
)
echo.

echo [4/7] Removing UV cache...
if exist "%USERPROFILE%\.cache\uv" (
    echo Removing UV cache...
    rmdir /s /q "%USERPROFILE%\.cache\uv" 2>nul
    echo UV cache removed.
) else (
    echo No UV cache found.
)
echo.

echo [5/7] Creating new virtual environment with UV...
uv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)
echo Virtual environment created.
echo.

echo [6/7] Activating virtual environment and installing dependencies...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

echo Installing project in editable mode...
uv pip install -e .
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)
echo Dependencies installed successfully.
echo.

echo [7/7] Setting up VS Code Python interpreter...
if not exist ".vscode" mkdir ".vscode"
echo { > ".vscode\settings.json"
echo   "python.defaultInterpreterPath": ".venv\\Scripts\\python.exe", >> ".vscode\settings.json"
echo   "python.terminal.activateEnvironment": true >> ".vscode\settings.json"
echo } >> ".vscode\settings.json"
echo VS Code settings configured.
echo.

echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Restart VS Code to pick up the new interpreter
echo 2. Run 'python main.py --help' to see available commands
echo 3. Run 'python test_setup.py' to verify the setup
echo.
echo Virtual environment: .venv
echo Python executable: .venv\Scripts\python.exe
echo.
pause
