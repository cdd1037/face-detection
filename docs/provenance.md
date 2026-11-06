# Model and source provenance

The managed implementation derives from the inference graph in
[cdd1037/face-detection](https://github.com/cdd1037/face-detection), which vendors
[ShiqiYu/libfacedetection](https://github.com/ShiqiYu/libfacedetection).
It is standalone at build/runtime, not an independent invention of the model.
The preserved upstream revision metadata is in
[UPSTREAM.txt](../licenses/libfacedetection/UPSTREAM.txt).

## Embedded model

`src/FaceDetection/Model/yunet.bin` was exported from the existing
`third_party/libfacedetection/src/facedetectcnn-data.cpp` in the source repository.
It contains 53 layers and 53,184 single-precision weights/biases, occupying 213,804
bytes including metadata. There was no retraining or quantization. The exporter
converts the source decimal literals to little-endian IEEE single precision,
validates dimensions and finite values, and preserves layer order. It serializes
fully before atomically replacing the destination, so invalid source data does not
leave a partially overwritten model.

The binary begins with `MFD1`, then a little-endian 32-bit layer count. Each layer
stores five 32-bit integers (input channels, output filters, depthwise flag, weight
count, bias count), followed by float weights and biases. The runtime validates
this against its fixed graph schema.

Maintainers can regenerate to a temporary file and compare before replacing it:

```sh
python3 tools/export_model.py /path/to/facedetectcnn-data.cpp /tmp/yunet-regenerated.bin
cmp src/FaceDetection/Model/yunet.bin /tmp/yunet-regenerated.bin
```

This is an explicit maintenance operation, not a build prerequisite. Keep source
revision metadata and notices with any future model update. A model update requires
separate correctness and quality evaluation, not merely a successful build.

## Licenses and notices

- Root `LICENSE` is the source repository's BSD-3-Clause license with its existing
  Copyright (c) 2026, cdd1037 notice. It is retained without inventing new authorship.
- libfacedetection's standalone `LICENSE` carries Copyright (c) 2015–2019, Shiqi Yu.
- Its inference source license header carries Copyright (c) 2018–2021, Shiqi Yu;
  it is retained as `SOURCE-NOTICE.txt`.
- The model data header carries Copyright (c) 2018–2023, Shiqi Yu; it is retained
  as `MODEL-NOTICE.txt`.

All of these notices apply together as appropriate; differing year ranges are
preserved from their sources rather than silently normalized. No Google Highway
source or binary is part of this library/package. The original native comparator
is a separate project with its own dependencies and notices.

The NASA portrait used by tests is documented as public domain by scikit-image.
Its exact source and transformation are recorded in `tests/fixtures/README.md`.
It is test data, not a package asset or a quality benchmark dataset.
