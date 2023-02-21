

Clear-Host
$current_dir=$PSScriptRoot
Import-Module "$current_dir\pmu.psm1" -Force

return

# CleanPmuOutputDirectory

D:\anaconda3\shell\condabin\conda-hook.ps1
conda activate base
#Scrap today
python scrap.py
#Predict today
python predicter.py
#retrieve resultat of yesterday
python resultat.py

LoadAllPmuIntoDatabase

