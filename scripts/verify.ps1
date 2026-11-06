$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$env:DOTNET_NOLOGO = '1'

function Invoke-DotNet {
    & dotnet @args
    if ($LASTEXITCODE -ne 0) { throw "dotnet failed with exit code $LASTEXITCODE" }
}

Invoke-DotNet build FaceDetection.sln -c Release '-m:1'
Invoke-DotNet run --project tests/FaceDetection.Tests -c Release --no-build
$oldAvx2 = $env:DOTNET_EnableAVX2
$oldIntrinsics = $env:DOTNET_EnableHWIntrinsic
try {
    $env:DOTNET_EnableAVX2 = '0'
    Invoke-DotNet run --project tests/FaceDetection.Tests -c Release --no-build
    $env:DOTNET_EnableAVX2 = $oldAvx2
    $env:DOTNET_EnableHWIntrinsic = '0'
    Invoke-DotNet run --project tests/FaceDetection.Tests -c Release --no-build
}
finally {
    $env:DOTNET_EnableAVX2 = $oldAvx2
    $env:DOTNET_EnableHWIntrinsic = $oldIntrinsics
}
Invoke-DotNet build FaceDetection.sln -c Release '-m:1' '-t:Rebuild' '-p:FaceDetectionScalar=true'
Invoke-DotNet run --project tests/FaceDetection.Tests -c Release --no-build
Invoke-DotNet build FaceDetection.sln -c Release '-m:1' '-t:Rebuild'
Invoke-DotNet run --project tests/FaceDetection.Tests -c Release --no-build
Invoke-DotNet pack src/FaceDetection -c Release --no-build -o artifacts/packages
