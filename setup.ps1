param(
    [string]$EnvName = "mbb-gs",
    [string]$PythonVersion = "3.10",
    [string]$CudaVersion = "12.6",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu126",
    [string]$CudaHome = "",
    [switch]$SkipEnvCreate,
    [switch]$SkipPackages,
    [switch]$SkipCudaBuild,
    [switch]$SkipSmokeMedia,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root


function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Message
    Write-Host "============================================================"
}


function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )

    Write-Host ">> $FilePath $($Arguments -join ' ')"
    & $FilePath @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw ("Command failed with exit code {0}: {1}" -f $LASTEXITCODE, $FilePath)
    }
}


function Find-CondaExe {
    $command = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
        (Join-Path $env:LOCALAPPDATA "miniconda3\Scripts\conda.exe")
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}


function Import-VsEnvironment {
    param([string]$VsDevCmd)

    $command = "call `"$VsDevCmd`" -arch=x64 -host_arch=x64 >nul && set"
    $environmentLines = & $env:ComSpec /d /c $command

    if ($LASTEXITCODE -ne 0) {
        throw "Could not initialize Visual Studio x64 environment."
    }

    foreach ($line in $environmentLines) {
        $index = $line.IndexOf("=")

        if ($index -le 0) {
            continue
        }

        $name = $line.Substring(0, $index)
        $value = $line.Substring($index + 1)

        Set-Item -Path "Env:$name" -Value $value
    }
}


Write-Host ""
Write-Host "MBB-GS Windows setup"
Write-Host "Repository: $Root"
Write-Host "Conda env:  $EnvName"
Write-Host "Python:     $PythonVersion"
Write-Host "CUDA:       $CudaVersion"


Write-Step "1. Detecting Conda"

$CondaExe = Find-CondaExe

if (-not $CondaExe) {
    throw "Conda was not found. Install Miniconda or Anaconda and run setup again."
}

Write-Host "Conda: $CondaExe"


Write-Step "2. Checking Conda environment"

& $CondaExe run -n $EnvName python --version *> $null
$EnvExists = $LASTEXITCODE -eq 0

if (-not $EnvExists) {
    if ($SkipEnvCreate) {
        throw "Environment $EnvName does not exist and SkipEnvCreate was specified."
    }

    Write-Host "Creating Conda environment $EnvName..."
    Invoke-Checked $CondaExe @(
        "create",
        "-y",
        "-n",
        $EnvName,
        "python=$PythonVersion"
    )
}
else {
    Write-Host "Environment $EnvName already exists."
}

Invoke-Checked $CondaExe @(
    "run",
    "-n",
    $EnvName,
    "python",
    "--version"
)


if (-not $SkipPackages) {
    Write-Step "3. Installing Python dependencies"

    Invoke-Checked $CondaExe @(
        "run",
        "-n",
        $EnvName,
        "python",
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
        "setuptools",
        "wheel",
        "ninja"
    )

    Invoke-Checked $CondaExe @(
        "run",
        "-n",
        $EnvName,
        "python",
        "-m",
        "pip",
        "install",
        "torch",
        "torchvision",
        "--index-url",
        $TorchIndexUrl
    )

    if (Test-Path (Join-Path $Root "requirements.txt")) {
        Invoke-Checked $CondaExe @(
            "run",
            "-n",
            $EnvName,
            "python",
            "-m",
            "pip",
            "install",
            "-r",
            (Join-Path $Root "requirements.txt")
        )
    }

    Invoke-Checked $CondaExe @(
        "run",
        "-n",
        $EnvName,
        "python",
        "-m",
        "pip",
        "install",
        "-e",
        $Root
    )
}
else {
    Write-Step "3. Python dependency installation skipped"
}


Write-Step "4. Detecting CUDA Toolkit"

if ([string]::IsNullOrWhiteSpace($CudaHome)) {
    $preferredCuda = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA\v$CudaVersion"

    if (Test-Path (Join-Path $preferredCuda "bin\nvcc.exe")) {
        $CudaHome = $preferredCuda
    }
    elseif ($env:CUDA_HOME -and (Test-Path (Join-Path $env:CUDA_HOME "bin\nvcc.exe"))) {
        $CudaHome = $env:CUDA_HOME
    }
    elseif ($env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH "bin\nvcc.exe"))) {
        $CudaHome = $env:CUDA_PATH
    }
    else {
        $nvccCommand = Get-Command nvcc.exe -ErrorAction SilentlyContinue

        if ($nvccCommand) {
            $CudaHome = Split-Path (Split-Path $nvccCommand.Source -Parent) -Parent
        }
    }
}

if ([string]::IsNullOrWhiteSpace($CudaHome)) {
    throw "CUDA Toolkit was not found."
}

$NvccPath = Join-Path $CudaHome "bin\nvcc.exe"

if (-not (Test-Path $NvccPath)) {
    throw "nvcc.exe was not found at $NvccPath"
}

$env:CUDA_HOME = $CudaHome
$env:CUDA_PATH = $CudaHome

if ($env:PATH -notlike "*$CudaHome\bin*") {
    $env:PATH = "$CudaHome\bin;$env:PATH"
}

Write-Host "CUDA_HOME: $CudaHome"
Invoke-Checked $NvccPath @("--version")


Write-Step "5. Checking FFmpeg"

$Ffmpeg = Get-Command ffmpeg.exe -ErrorAction SilentlyContinue
$Ffprobe = Get-Command ffprobe.exe -ErrorAction SilentlyContinue

if (-not $Ffmpeg) {
    throw "ffmpeg.exe was not found in PATH."
}

if (-not $Ffprobe) {
    throw "ffprobe.exe was not found in PATH."
}

Write-Host "FFmpeg:  $($Ffmpeg.Source)"
Write-Host "FFprobe: $($Ffprobe.Source)"


if (-not $SkipCudaBuild) {
    Write-Step "6. Detecting Visual Studio C++ toolchain"

    $VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"

    if (-not (Test-Path $VsWhere)) {
        throw "vswhere.exe was not found. Install Visual Studio 2022 C++ build tools."
    }

    $VsInstall = & $VsWhere `
        -latest `
        -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath | Select-Object -First 1

    if ([string]::IsNullOrWhiteSpace($VsInstall)) {
        throw "Visual Studio C++ x64 tools were not found."
    }

    $VsInstall = $VsInstall.Trim()
    $VsDevCmd = Join-Path $VsInstall "Common7\Tools\VsDevCmd.bat"

    if (-not (Test-Path $VsDevCmd)) {
        throw "VsDevCmd.bat was not found at $VsDevCmd"
    }

    Write-Host "Visual Studio: $VsInstall"

    Import-VsEnvironment $VsDevCmd

    $env:DISTUTILS_USE_SDK = "1"

    $ClCommand = Get-Command cl.exe -ErrorAction SilentlyContinue

    if (-not $ClCommand) {
        throw "cl.exe is still unavailable after initializing Visual Studio."
    }

    Write-Host "MSVC: $($ClCommand.Source)"


    Write-Step "7. Building raster_cuda"

    Push-Location (Join-Path $Root "cuda\raster_cuda")
    try {
        Invoke-Checked $CondaExe @(
            "run",
            "-n",
            $EnvName,
            "python",
            "setup.py",
            "build_ext",
            "--inplace"
        )
    }
    finally {
        Pop-Location
    }


    Write-Step "8. Building gabor_audio_cuda"

    Push-Location (Join-Path $Root "cuda\gabor_audio_cuda")
    try {
        Invoke-Checked $CondaExe @(
            "run",
            "-n",
            $EnvName,
            "python",
            "setup.py",
            "build_ext",
            "--inplace"
        )
    }
    finally {
        Pop-Location
    }
}
else {
    Write-Step "6-8. CUDA build skipped"
}


if (-not $SkipSmokeMedia) {
    Write-Step "9. Generating synthetic smoke media"

    Invoke-Checked $CondaExe @(
        "run",
        "-n",
        $EnvName,
        "python",
        (Join-Path $Root "scripts\generate_smoke_media.py"),
        "--force"
    )
}
else {
    Write-Step "9. Smoke media generation skipped"
}


Write-Step "10. Running environment doctor"

$DoctorArguments = @(
    "run",
    "-n",
    $EnvName,
    "python",
    (Join-Path $Root "scripts\doctor.py")
)

if (-not $SkipTests) {
    $DoctorArguments += "--tests"
}

Invoke-Checked $CondaExe $DoctorArguments


Write-Host ""
Write-Host "============================================================"
Write-Host "MBB-GS setup completed successfully"
Write-Host "============================================================"
Write-Host ""
Write-Host "Activate the environment with:"
Write-Host "conda activate $EnvName"
Write-Host ""
Write-Host "Run the doctor with:"
Write-Host "python scripts\doctor.py --tests"
Write-Host ""
Write-Host "Run the smoke pipeline with:"
Write-Host "python scripts\pipeline\run_pipeline_video_audio.py --config configs\examples\smoke_20frames\pipeline.json"
Write-Host ""
