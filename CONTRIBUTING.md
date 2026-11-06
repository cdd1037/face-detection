# Contributing

Use the .NET 8 SDK. Build and run the console test suite as described in
[testing](docs/testing.md); exercise both default and scalar configurations.
There are no third-party runtime dependencies or native library build steps.
The SDK may restore Microsoft analyzer/AOT build tools.

Keep changes focused and preserve the public `FaceDetection` namespace and
API unless an explicit API change is being reviewed. Keep image codecs, engine
bindings, native comparison tools, and application scheduling outside the library.
Add regression tests for new behavior, especially layout validation, workspace
ownership, numerical boundaries, and kernel bounds/tails.

Do not alter model bytes, graph order, or arithmetic as incidental cleanup. Explain
intentional numerical changes and validate outputs and performance separately.
Retain original source/model notices. Follow [provenance](docs/provenance.md) when
regenerating weights; no model conversion runs during consumer builds.

Document which hardware and runtime you actually tested. CI definitions and
portable fallbacks are not proof of unrun platform support. Keep benchmark results
reproducible and concise; raw logs, build products, experimental source snapshots,
and one-off delivery reports do not belong in the maintained source tree.

Use local project references or a locally packed feed for consumer checks. Do not
assume the package is published or upload/release artifacts as part of validation.
