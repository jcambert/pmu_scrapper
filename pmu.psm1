
$current_dir=$PSScriptRoot
$base_url="https://localhost:44381/api/pmu/"

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
        [ValidateNotNullOrEmpty()][string]$url,
        [string] $path="."
        );
    process{
            $request_body=(@{
                "filename"="$current_dir\output\$path\$Filename"
            }|ConvertTo-Json)

            $request_url="$base_url$url"
            Invoke-WebRequest -Method 'Post' -Uri $request_url -Body $request_body -ContentType "application/json"

        }
    }
function LoadAllPmuIntoDatabase{
    param(
        [srting]$path
    )
    process{
        LoadPmuIntoDatabase -Path $path -Filename "predicted.csv" -Url "LoadPredictedIntoDb"
        LoadPmuIntoDatabase -Path $path -Filename "resultats_plat.csv" -Url "LoadResultatIntoDb"
        LoadPmuIntoDatabase -Path $path -Filename "resultats_trot_attele.csv" -Url "LoadResultatIntoDb"
        LoadPmuIntoDatabase -Path $path -Filename "resultats_trot_monte.csv" -Url "LoadResultatIntoDb"
        LoadPmuIntoDatabase -Path $path -Filename "resultats_obstacle.csv" -Url "LoadResultatIntoDb"
        LoadPmuIntoDatabase -Path $path -Filename "courses.csv" -Url "LoadCourseIntoDb"

    }
}

function Invoke-LoadPmu{
    [CmdletBinding()]
    param(
        [string]$start,
        [srting]$path
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

            LoadAllPmuIntoDatabase -Path $path
        }
}

Set-Alias ldpmu Invoke-LoadPmu
Export-ModuleMember -Function Invoke-LoadPmu,CleanPmuOutputDirectory,LoadPmuIntoDatabase, LoadAllPmuIntoDatabase -Alias ldpmu