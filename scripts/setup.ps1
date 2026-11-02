$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$third = Join-Path $root "third_party"
New-Item -ItemType Directory -Force -Path $third | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $root "models") | Out-Null

$cmakeExe = Join-Path $third "cmake-3.31.6-windows-x86_64\bin\cmake.exe"
if (-not (Test-Path $cmakeExe)) {
    $zip = Join-Path $third "cmake.zip"
    if (-not (Test-Path $zip)) {
        Write-Host "Downloading CMake..."
        curl.exe -L --fail --retry 3 -o $zip "https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-windows-x86_64.zip"
    }
    Write-Host "Extracting CMake..."
    Expand-Archive -Path $zip -DestinationPath $third -Force
}

$opencvConfig = Join-Path $third "opencv\build\OpenCVConfig.cmake"
if (-not (Test-Path $opencvConfig)) {
    $exe = Join-Path $third "opencv.exe"
    if (-not (Test-Path $exe)) {
        Write-Host "Downloading OpenCV..."
        curl.exe -L --fail --retry 3 -o $exe "https://github.com/opencv/opencv/releases/download/4.11.0/opencv-4.11.0-windows.exe"
    }
    Write-Host "Extracting OpenCV..."
    & $exe -o"$third" -y
}

$model = Join-Path $root "models\yolov8n-pose.onnx"
if (-not (Test-Path $model)) {
    Write-Host "Downloading pose model..."
    curl.exe -L --fail --retry 3 -o $model "https://github.com/CVHub520/X-AnyLabeling/releases/download/v1.0.0/yolov8n-pose.onnx"
}

Write-Host "Setup complete."
