// swift-tools-version: 6.3
import PackageDescription

let package = Package(name: "WeeToddMOSSImage", platforms: [.macOS(.v14)],
  products: [.library(name: "MOSSImageMLX", targets: ["MOSSImageMLX"]),
    .executable(name: "WeeToddMOSSImageWorker", targets: ["WeeToddMOSSImageWorker"])],
  dependencies: [
    .package(path: "../weetodd-image-layout"), .package(path: "../weetodd-inference"),
    .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.32.3"),
    .package(url: "https://github.com/ml-explore/mlx-swift-lm", exact: "3.32.3", traits: []),
    .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.1.9"),
  ], targets: [
    .target(name: "MOSSImageMLX", dependencies: [.product(name: "ImageLayout", package: "weetodd-image-layout"),
      .product(name: "TensorIO", package: "weetodd-inference"), .product(name: "MLX", package: "mlx-swift"),
      .product(name: "MLXNN", package: "mlx-swift"), .product(name: "MLXFast", package: "mlx-swift"),
      .product(name: "MLXRandom", package: "mlx-swift"), .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
      .product(name: "MLXVLM", package: "mlx-swift-lm"), .product(name: "MLXGuidedGeneration", package: "mlx-swift-lm"),
      .product(name: "Tokenizers", package: "swift-transformers")]),
    .executableTarget(name: "WeeToddMOSSImageWorker", dependencies: ["MOSSImageMLX"],
      swiftSettings: [.unsafeFlags(["-parse-as-library"])]),
    .testTarget(name: "MOSSImageMLXTests", dependencies: ["MOSSImageMLX"]),
  ], swiftLanguageModes: [.v6])
