#!/bin/bash
# Native Swift numerical tests, isolated from the video engines' MLX versions.
set -euo pipefail
package_root="$(cd "$(dirname "$0")/.." && pwd)"
swift build --build-system native --package-path "$package_root" -c release --build-tests --jobs 2 -Xswiftc -enable-testing -Xcxx -DNDEBUG
binary_root="$(swift build --build-system native --package-path "$package_root" -c release --show-bin-path)"
bash "$package_root/scripts/build_mlx_metallib.sh" "$binary_root"
swift test --build-system native --package-path "$package_root" -c release --skip-build
