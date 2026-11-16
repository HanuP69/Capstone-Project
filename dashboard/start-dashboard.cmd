@echo off
setlocal
cd /d "%~dp0"

if not exist "node_modules\vite\bin\vite.js" (
  echo Dashboard dependencies are missing. Installing with normal npm TLS verification...
  call npm.cmd install
  if errorlevel 1 (
    echo Dependency installation failed. Review the npm error above and try again.
    pause
    exit /b 1
  )
)

echo Starting Capstone Project on http://127.0.0.1:5173/verify
start "Capstone Project - Vite server" /D "%~dp0" cmd.exe /k "npm.cmd run dev -- --strictPort"

powershell.exe -NoProfile -Command "$deadline=(Get-Date).AddSeconds(30); do { try { $r=Invoke-WebRequest -Uri 'http://127.0.0.1:5173/verify' -TimeoutSec 1; if ($r.StatusCode -eq 200) { exit 0 } } catch { }; Start-Sleep -Milliseconds 500 } while ((Get-Date) -lt $deadline); exit 1"
if errorlevel 1 (
  echo The local page did not respond within 30 seconds. Check the Vite server window for errors.
  pause
  exit /b 1
)

start "" "http://127.0.0.1:5173/verify"
echo The dashboard is open. To stop the server, press Ctrl+C in its Vite console window.
endlocal
