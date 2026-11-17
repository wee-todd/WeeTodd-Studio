#!/bin/bash
# Native numerical/contract checks with matching pinned Metal resources.
# Model weights and external-reference qualification remain opt-in.
set -euo pipefail
package_root="$(cd "$(dirname "$0")/.." && pwd)"
swift build --build-system native --package-path "$package_root" -c release --build-tests --jobs 2 -Xswiftc -enable-testing -Xcxx -DNDEBUG
binary_dir="$(swift build --build-system native --package-path "$package_root" -c release --show-bin-path)"
"$package_root/../weetodd-mlx/scripts/build_mlx_metallib.sh" "$binary_dir" "$package_root/.build/checkouts/mlx-swift"
test_binary_dir="$binary_dir/WeeToddMOSSImagePackageTests.xctest/Contents/MacOS"
mkdir -p "$test_binary_dir"
cp "$binary_dir/mlx.metallib" "$test_binary_dir/mlx.metallib"
swift test --build-system native --package-path "$package_root" -c release --skip-build
