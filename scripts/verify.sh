#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export DOTNET_NOLOGO=1

dotnet build FaceDetection.sln -c Release -m:1
dotnet run --project tests/FaceDetection.Tests -c Release --no-build
DOTNET_EnableAVX2=0 dotnet run --project tests/FaceDetection.Tests -c Release --no-build
DOTNET_EnableHWIntrinsic=0 dotnet run --project tests/FaceDetection.Tests -c Release --no-build
# Rebuild explicitly so switching compile-time constants cannot reuse another variant.
dotnet build FaceDetection.sln -c Release -m:1 -t:Rebuild -p:FaceDetectionScalar=true
dotnet run --project tests/FaceDetection.Tests -c Release --no-build
dotnet build FaceDetection.sln -c Release -m:1 -t:Rebuild
dotnet run --project tests/FaceDetection.Tests -c Release --no-build
dotnet pack src/FaceDetection -c Release --no-build -o artifacts/packages
