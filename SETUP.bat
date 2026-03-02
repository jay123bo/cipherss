@echo off
title Cipher.AI — Setup
color 0A
cls

echo.
echo  ========================================
echo    CIPHER.AI — PERSONAL AI SETUP
echo  ========================================
echo.
echo  This will set up YOUR own AI on your PC.
echo  No internet needed after setup. 100%% yours.
echo.
pause

:: ── STEP 1: Check Python ────────────────────────────────────────
cls
echo.
echo  [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
  echo.
  echo  ERROR: Python is not installed!
  echo.
  echo  Please do this:
  echo  1. Go to https://www.python.org/downloads
  echo  2. Click the big yellow "Download Python" button
  echo  3. Run the installer
  echo  4. IMPORTANT: Check "Add Python to PATH" at the bottom!
  echo  5. Come back and double-click this file again
  echo.
  pause
  exit
)
echo  Python found!

:: ── STEP 2: Install libraries ───────────────────────────────────
echo.
echo  [2/4] Installing AI libraries (this may take a few minutes)...
echo.
pip install torch --quiet
pip install fastapi uvicorn --quiet
echo  Done!

:: ── STEP 3: Check for data.txt ──────────────────────────────────
cls
echo.
echo  [3/4] Checking for training data...
echo.
if not exist data.txt (
  echo  You need a file called data.txt in this folder!
  echo.
  echo  What to put in it:
  echo  - Copy/paste text from books, articles, anything
  echo  - More text = smarter AI
  echo  - Aim for at least 100,000 characters
  echo.
  echo  TIP: Go to gutenberg.org and copy a free book!
  echo.
  echo  Once you have data.txt, double-click this file again.
  echo.
  pause
  exit
)

:: Count characters roughly
for %%A in (data.txt) do set SIZE=%%~zA
echo  data.txt found! Size: %SIZE% bytes
echo.
if %SIZE% LSS 10000 (
  echo  WARNING: Your data.txt is pretty small.
  echo  Your AI might not be very smart.
  echo  Consider adding more text for better results.
  echo.
  pause
)

:: ── STEP 4: Train ───────────────────────────────────────────────
cls
echo.
echo  [4/4] Starting AI training...
echo.
echo  ========================================
echo   This will take 30min - 2 hours.
echo   Leave this window open!
echo   Your AI saves automatically as it trains.
echo  ========================================
echo.
pause

python train.py

:: ── DONE ────────────────────────────────────────────────────────
cls
echo.
echo  ========================================
echo   YOUR AI IS READY!
echo  ========================================
echo.
echo  To chat with your AI:
echo  1. Double-click "CHAT WITH MY AI.bat"
echo.
echo  Your AI is saved in: my_ai.pt
echo  This file = your AI brain. Keep it safe!
echo.
pause
