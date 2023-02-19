
$current_dir=Get-Location
$base_url="https://localhost:44381/api/app/pmu/"

#Invoke-WebRequest -Method 'Post' -Uri $url -Body ($body|ConvertTo-Json) -ContentType "application/json"
function CleanPmuOutputDirectory{
    process{
        Remove-Item "$current_dir\output\*.*"
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

Export-ModuleMember -Function CleanPmuOutputDirectory,LoadPmuIntoDatabase, LoadAllPmuIntoDatabase