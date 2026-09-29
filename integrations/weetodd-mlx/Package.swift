// swift-tools-version: 6.3
import PackageDescription
let package = Package(
  name: "WeeToddMLX", platforms: [.macOS(.v14)],
  products: [.executable(name: "WeeToddH3MLXWorker", targets: ["WeeToddH3MLXWorker"]), .executable(name: "WeeToddMLXDecodeProbe", targets: ["WeeToddMLXDecodeProbe"]), .executable(name: "WeeToddMLXPipelineProbe", targets: ["WeeToddMLXPipelineProbe"]), .executable(name: "WeeToddMLXDenoiserProbe", targets: ["WeeToddMLXDenoiserProbe"]), .executable(name: "WeeToddMLXStackProbe", targets: ["WeeToddMLXStackProbe"]), .executable(name: "WeeToddMLXBlockProbe", targets: ["WeeToddMLXBlockProbe"]), .library(name: "LTX25MLX", targets: ["LTX25MLX"]), .library(name: "H3MLX", targets: ["H3MLX"])],
  dependencies: [
    .package(path: "../weetodd-inference"),
    .package(url: "https://github.com/ml-explore/mlx-swift.git", revision: "901941965d82e4a216d4d117231d847d194c563d"),
  ],
  targets: [
    .executableTarget(name: "WeeToddH3MLXWorker", dependencies: ["H3MLX",
      .product(name: "InferenceContracts", package: "weetodd-inference"),
      .product(name: "InferenceMedia", package: "weetodd-inference"),
      .product(name: "MLX", package: "mlx-swift")]),
    .executableTarget(name: "WeeToddLTXWorker", dependencies: ["LTX25MLX",
      .product(name: "InferenceContracts", package: "weetodd-inference")]),
    .executableTarget(name: "WeeToddMLXDecodeProbe", dependencies: ["LTX25MLX"]),
    .executableTarget(name: "WeeToddMLXPipelineProbe", dependencies: ["LTX25MLX"]),
    .executableTarget(name: "WeeToddMLXDenoiserProbe", dependencies: ["LTX25MLX"]),
    .executableTarget(name: "WeeToddMLXStackProbe", dependencies: ["LTX25MLX"]),
    .executableTarget(name: "WeeToddMLXBlockProbe", dependencies: ["LTX25MLX"]),
    .target(name: "LTX25MLX", dependencies: [
      .product(name: "InferenceContracts", package: "weetodd-inference"),
      .product(name: "TensorIO", package: "weetodd-inference"),
      .product(name: "LTX25Engine", package: "weetodd-inference"),
      .product(name: "LTX25Text", package: "weetodd-inference"),
      .product(name: "LTX25Video", package: "weetodd-inference"),
      .product(name: "LTX25Audio", package: "weetodd-inference"),
      .product(name: "InferenceMedia", package: "weetodd-inference"),
      .product(name: "AdapterRuntime", package: "weetodd-inference"),
      .product(name: "MLX", package: "mlx-swift")]),
    .target(name: "H3MLX", dependencies: [
      .product(name: "TensorIO", package: "weetodd-inference"),
      .product(name: "InferenceContracts", package: "weetodd-inference"),
      .product(name: "MLX", package: "mlx-swift"),
      .product(name: "MLXRandom", package: "mlx-swift"),
      .product(name: "MLXNN", package: "mlx-swift")]),
    .testTarget(name: "H3MLXTests", dependencies: ["H3MLX"]),
    .target(name: "InferenceTestSupport", path: "Tests/Support"),
    .testTarget(name: "LTX25MLXTests", dependencies: ["LTX25MLX", "InferenceTestSupport"], resources: [.copy("Fixtures")]),
  ], swiftLanguageModes: [.v6])
