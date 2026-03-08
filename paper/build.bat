@echo off
cd /d "%~dp0"
C:\texlive\2025\bin\windows\xelatex.exe -interaction=nonstopmode main.tex
C:\texlive\2025\bin\windows\xelatex.exe -interaction=nonstopmode main.tex
echo.
echo Done. Check main.pdf
pause
