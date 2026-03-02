@echo off
title Cipher.AI — Chat
color 0A
cls

echo.
echo  ========================================
echo    CIPHER.AI — YOUR PERSONAL AI
echo  ========================================
echo.

if not exist my_ai.pt (
  echo  ERROR: Your AI hasnt been trained yet!
  echo  Run SETUP.bat first.
  echo.
  pause
  exit
)

echo  Starting your AI...
echo  Type your message and press Enter to chat.
echo  Type "quit" to exit.
echo.
python chat.py
pause
