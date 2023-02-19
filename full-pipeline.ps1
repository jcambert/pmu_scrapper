Clear-Host
$current_dir=Get-Location
Import-Module "$current_dir\update-database.psm1" -Force

# CleanPmuOutputDirectory

D:\anaconda3\shell\condabin\conda-hook.ps1
conda activate base
python __init__.py
python predicter.py
python resultat.py

LoadAllPmuIntoDatabase

