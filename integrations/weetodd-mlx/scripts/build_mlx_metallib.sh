#!/bin/bash
# SwiftPM does not compile Metal resources. Build the non-JIT kernels from the
# exact resolved MLX source; the remainder are supplied by MLX's JIT backend.
set -euo pipefail
package_root="$(cd "$(dirname "$0")/.." && pwd)"
mlx_root="$package_root/.build/checkouts/mlx-swift/Source/Cmlx/mlx"
output_dir="${1:-$package_root/.build/arm64-apple-macosx/release}"
test -f "$mlx_root/mlx/backend/metal/kernels/rms_norm.metal"
mkdir -p "$output_dir/mlx-air"
kernel_objects=()
for kernel in arg_reduce conv dot layer_norm random rms_norm rope scaled_dot_product_attention fence; do
  metal_standard="metal3.1"
  if [[ "$kernel" == "fence" ]]; then metal_standard="metal3.2"; fi
  # XcodeDefault pins Swift but hides Apple's separately installed Metal toolchain.
  # Keep the selected Xcode/SDK while allowing xcrun to discover the real compiler.
  env -u TOOLCHAINS xcrun -sdk macosx metal -std="$metal_standard" -x metal -fno-fast-math -Wno-c++17-extensions -Wno-c++20-extensions \
    -mmacosx-version-min=14.0 -I "$mlx_root" \
    -c "$mlx_root/mlx/backend/metal/kernels/$kernel.metal" -o "$output_dir/mlx-air/$kernel.air"
  kernel_objects+=("$output_dir/mlx-air/$kernel.air")
done
env -u TOOLCHAINS xcrun -sdk macosx metallib "${kernel_objects[@]}" -o "$output_dir/mlx.metallib"
# XCTest loads the package binary from inside this bundle rather than the CLI
# directory. Keep the library beside that binary as required by MLX discovery.
test_binary_dir="$output_dir/WeeToddMLXPackageTests.xctest/Contents/MacOS"
mkdir -p "$test_binary_dir"
cp "$output_dir/mlx.metallib" "$test_binary_dir/mlx.metallib"
