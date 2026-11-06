# Public-domain integration fixture

`astronaut.rgb`: 256 × 256, top-down RGB24, stride 768 bytes.
NASA portrait of Eileen Collins, downloaded from
https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/astronaut.png
and resized from 512 × 512 with Pillow's Lanczos resampler; no crop or framing rules.

scikit-image documents this NASA image as public domain:
https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/_fetchers.py#L355-L373
Original: https://www.flickr.com/photos/nasacommons/16511503260/

This fixture is only used by tests and is never included in NuGet packages.
