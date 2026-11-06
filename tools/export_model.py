#!/usr/bin/env python3
"""One-time maintainer conversion of libfacedetection's BSD-licensed C float arrays.
No C++ compiler is used; the generated little-endian binary is embedded by the C# project.
Usage: python3 tools/export_model.py /path/to/facedetectcnn-data.cpp [output.bin]
"""
import argparse
import io
import math
import os
import tempfile
from pathlib import Path
import re
import struct

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', type=Path)
parser.add_argument('output', nargs='?', type=Path, default=Path(__file__).resolve().parents[1] / 'src/FaceDetection/Model/yunet.bin')
args = parser.parse_args()
source = args.source.read_text(encoding='utf-8')
arrays = {}
for name, dimensions, data in re.findall(r'float\s+(\w+)\[([\d*]+)\]\s*=\s*\{([^}]+)\}', source):
    values = [float(value.strip().removesuffix('f')) for value in data.split(',') if value.strip()]
    count = 1
    for dimension in dimensions.split('*'):
        count *= int(dimension)
    if count != len(values):
        raise ValueError(f'{name}: declared {count}, found {len(values)}')
    arrays[name] = values
layers = re.findall(r'\{(\d+),\s*(\d+),\s*(true|false),\s*(true|false),\s*(true|false),\s*(\w+),\s*(\w+)\}', source)
if len(layers) != 53:
    raise ValueError(f'Expected 53 layers, found {len(layers)}')
# Validate and serialize completely before replacing any checked-in model.
with io.BytesIO() as out:
    out.write(b'MFD1')
    out.write(struct.pack('<i', len(layers)))
    total = 0
    for channels, filters, depthwise, pointwise, relu, weights_name, biases_name in layers:
        channels, filters = int(channels), int(filters)
        dw, pw = depthwise == 'true', pointwise == 'true'
        if dw == pw or dw and channels != filters:
            raise ValueError('Unsupported layer')
        weights, biases = arrays[weights_name], arrays[biases_name]
        if len(weights) != channels * (9 if dw else filters) or len(biases) != filters:
            raise ValueError(f'Unexpected sizes for {weights_name}')
        if not all(math.isfinite(value) for value in weights + biases):
            raise ValueError(f'Non-finite values in {weights_name} or {biases_name}')
        out.write(struct.pack('<5i', channels, filters, int(dw), len(weights), len(biases)))
        out.write(struct.pack(f'<{len(weights)}f', *weights))
        out.write(struct.pack(f'<{len(biases)}f', *biases))
        total += len(weights) + len(biases)
    payload = out.getvalue()

args.output.parent.mkdir(parents=True, exist_ok=True)
# A sibling temporary file keeps replacement atomic on the destination filesystem.
fd, temporary = tempfile.mkstemp(prefix=args.output.name + '.', suffix='.tmp', dir=args.output.parent)
try:
    with os.fdopen(fd, 'wb') as out:
        out.write(payload)
    os.replace(temporary, args.output)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)

print(f'Exported {len(layers)} layers, {total} floats, {args.output.stat().st_size} bytes to {args.output}')
