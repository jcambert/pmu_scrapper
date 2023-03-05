
$current_dir=$PSScriptRoot
$base_url="https://localhost:44381/api/app/pmu/"

#Invoke-WebRequest -Method 'Post' -Uri $url -Body ($body|ConvertTo-Json) -ContentType "application/json"
function CleanPmuOutputDirectory{
    process{
        Remove-Item "$current_dir\output\*.csv"
    }
}

function CleanPmuInputDirectory{
    process{
        Remove-Item "$current_dir\input\*.csv"
    }
}
function LoadPmuIntoDatabase{
    [CmdletBinding()]
    param(
        [ValidateNotNullOrEmpty()][string]$Filename,
        [ValidateNotNullOrEmpty()][string]$url
        );
    process{
            $request_body=(@{
                "filename"="$current_dir\output\$Filename"
            }|ConvertTo-Json)

            $request_url="$base_url$url"
            Invoke-WebRequest -Method 'Post' -Uri $request_url -Body $request_body -ContentType "application/json"

        }
    }
function LoadAllPmuIntoDatabase{
    process{
        LoadPmuIntoDatabase -Filename "predicted.csv" -Url "load-predicted-into-db"
        LoadPmuIntoDatabase -Filename "resultats_plat.csv" -Url "load-resultat-into-db"
        LoadPmuIntoDatabase -Filename "resultats_trot_attele.csv" -Url "load-resultat-into-db"
        LoadPmuIntoDatabase -Filename "resultats_trot_monte.csv" -Url "load-resultat-into-db"
        LoadPmuIntoDatabase -Filename "resultats_obstacle.csv" -Url "load-resultat-into-db"

    }
}

function Invoke-LoadPmu{
    [CmdletBinding()]
    param(
        [string]$start
        )
        begin{
            
        }
        process{
            Clear-Host
            Set-Location $current_dir
            CleanPmuInputDirectory
            [System.DateOnly]$now   =[System.DateOnly]::MinValue
            if(![System.DateOnly]::TryParse($start,[ref] $now)){
                $dt=Get-Date
                $now = [System.DateOnly]::FromDateTime($dt)
                $today= $now.ToString( "ddMMyyyy")
            }else{
                $today= $now.ToString( "ddMMyyyy")
            }
            $yesterday = $now.AddDays(-1).ToString("ddMMyyyy")

            # $drive = $PSScriptRoot | Split-Path  -Qualifier
            # "$drive\anaconda3\\shell\condabin\conda-hook.ps1"
            
            conda activate base
            # Write-Output "start=$today"
            # Write-Output "yesterday=$yesterday"
            # return
            #Scrap today
            python scrap.py start=$today count=1
            #Predict today
            python predicter.py 
            #retrieve resultat of yesterday
            python resultat.py start=$yesterday count=1

            LoadAllPmuIntoDatabase
        }
}

Set-Alias ldpmu Invoke-LoadPmu
Export-ModuleMember -Function Invoke-LoadPmu,CleanPmuOutputDirectory,LoadPmuIntoDatabase, LoadAllPmuIntoDatabase -Alias ldpmu