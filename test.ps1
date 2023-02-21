Clear-Host
$now = Get-Date
$today= $now.ToString( "ddMMyyyy")
$yesterday=$now.AddDays(-1).ToString("ddMMyyyy")
Write-Output "Today: $today"
Write-Output "Yesterday: $yesterday"

# Write-Output $PSScriptRoot
# Write-Output $args
Write-Output @PSBoundParameters

