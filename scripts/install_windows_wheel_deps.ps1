$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string] $FilePath,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]] $Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE"
    }
}

$MsmPiUrl = "https://download.microsoft.com/download/7/2/7/72731ebb-b63c-4170-ade7-836966263a8f/msmpisetup.exe"
$MsmPiInstaller = Join-Path $env:TEMP "msmpisetup.exe"
Invoke-WebRequest -Uri $MsmPiUrl -OutFile $MsmPiInstaller
$MsmPiInstall = Start-Process -FilePath $MsmPiInstaller -ArgumentList "-unattend" -Wait -PassThru
if ($MsmPiInstall.ExitCode -ne 0) {
    throw "$MsmPiInstaller failed with exit code $($MsmPiInstall.ExitCode)"
}
if (-not (Test-Path "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe")) {
    throw "MS-MPI redistributable install did not produce mpiexec.exe"
}

if (-not (Test-Path "C:\vcpkg")) {
    Invoke-Checked git clone --depth 1 https://github.com/microsoft/vcpkg C:\vcpkg
}

Invoke-Checked C:\vcpkg\bootstrap-vcpkg.bat

$packages = @(
    "eigen3:x64-windows",
    "hdf5[core,cpp,zlib]:x64-windows",
    "msmpi:x64-windows"
)
$MaxAttempts = 5

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    & C:\vcpkg\vcpkg.exe install @packages --disable-metrics
    if ($LASTEXITCODE -eq 0) {
        exit 0
    }

    if ($attempt -eq $MaxAttempts) {
        throw "vcpkg install failed after $MaxAttempts attempts"
    }

    $delaySeconds = 30 * $attempt
    Write-Host "vcpkg install failed with exit code $LASTEXITCODE; retrying in $delaySeconds seconds ($attempt/$MaxAttempts)"
    Start-Sleep -Seconds $delaySeconds
}
