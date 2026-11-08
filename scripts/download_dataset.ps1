$ErrorActionPreference = "Stop"
$ProgressPreference = 'SilentlyContinue'   # much faster downloads in PowerShell

$doi = "doi:10.7910/DVN/EGZHFV"
$out = "E:\Sem5\cn\project\RL_6g\data\raw"
New-Item -ItemType Directory -Force -Path $out | Out-Null

# 1. Get the list of files
$url  = "https://dataverse.harvard.edu/api/datasets/:persistentId/versions/:latest-published/files?persistentId=$doi&limit=200"
$list = (Invoke-RestMethod $url).data
Write-Host "Files found:" $list.Count

# 2. Check free space (needs about 20 GB)
$totalGB = [math]::Round(($list | ForEach-Object { $_.dataFile.filesize } | Measure-Object -Sum).Sum / 1GB, 1)
$freeGB  = [math]::Round((Get-PSDrive E).Free / 1GB, 1)
Write-Host "Total size: $totalGB GB | Free on E: $freeGB GB"
if ($freeGB -lt ($totalGB + 2)) { throw "Not enough free space on E:" }

# 3. Download each file, skipping ones already complete, retrying on failure
$i = 0
foreach ($f in $list) {
    $i++
    $name = $f.dataFile.filename
    $id   = $f.dataFile.id
    $size = $f.dataFile.filesize
    $path = Join-Path $out $name

    if ((Test-Path $path) -and ((Get-Item $path).Length -eq $size)) {
        Write-Host "[$i/$($list.Count)] $name already done, skipping"
        continue
    }

    for ($try = 1; $try -le 3; $try++) {
        try {
            Write-Host "[$i/$($list.Count)] Downloading $name (try $try)"
            Invoke-WebRequest "https://dataverse.harvard.edu/api/access/datafile/$id" -OutFile "$path.part"
            Move-Item "$path.part" $path -Force
            break
        } catch {
            Write-Host "  failed: $($_.Exception.Message)"
            if (Test-Path "$path.part") { Remove-Item "$path.part" }
            if ($try -eq 3) { Write-Host "  GIVING UP on $name, run the script again later" }
        }
    }
}
Write-Host "Done."