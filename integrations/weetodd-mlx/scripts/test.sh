#!/bin/bash
# Separate executables prevent MLX/NNC metal-cpp symbol collisions. No Python
# runtime, model downloads, app replacement or environment installation.
set -euo pipefail
package_root="$(cd "$(dirname "$0")/.." && pwd)"
# Unlike `swift test`, release `swift build --build-tests` does not enable
# @testable imports automatically. Keep internal numerical primitives testable
# without making them part of the public inference API.
swift build --package-path "$package_root" -c release --build-tests --jobs 2 -Xswiftc -enable-testing
"$package_root/scripts/build_mlx_metallib.sh"
swift test --package-path "$package_root" -c release --skip-build
