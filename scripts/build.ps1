$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$cmake = Join-Path $root "third_party\cmake-3.31.6-windows-x86_64\bin\cmake.exe"
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $cmake) -or -not (Test-Path (Join-Path $root "third_party\opencv\build\OpenCVConfig.cmake"))) {
    & (Join-Path $PSScriptRoot "setup.ps1")
}
$build = Join-Path $root "build"
cmd.exe /c "call `"$vcvars`" && `"$cmake`" -S `"$root`" -B `"$build`" -G `"NMake Makefiles`" -DCMAKE_BUILD_TYPE=Release && `"$cmake`" --build `"$build`""
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Built build\hand_signal_car.exe"
