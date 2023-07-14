$Date = Get-Date -Format "yyyyMMdd"
$ArchiveName = "so-vits-svc-5.0-$Date.zip"

$ExcludeItems = @("__pycache__/", "*_pretrain/", "chkpt/", "data_svc/", "dataset_raw/", "files/", "logs/", "*git/")
$ExcludeArgs = $ExcludeItems | ForEach-Object { "-xr!$_" }

$Command = "7za a $ArchiveName $ExcludeArgs ./"

Invoke-Expression $Command