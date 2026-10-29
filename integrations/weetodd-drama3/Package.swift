// swift-tools-version: 6.2
import PackageDescription

let package = Package(
  name: "WeeToddDrama3", platforms: [.macOS(.v14)],
  products: [
    .library(name: "Drama3Contracts", targets: ["Drama3Contracts"]),
    .library(name: "Drama3MLX", targets: ["Drama3MLX"]),
    .executable(name: "WeeToddDrama3Worker", targets: ["WeeToddDrama3Worker"]),
  ],
  dependencies: [
    .package(
      url: "https://github.com/Blaizzy/mlx-audio-swift.git",
      revision: "244458d7b61e54ff135073ddc9ed76782bb20758")
  ],
  targets: [
    .target(name: "Drama3Contracts"),
    .target(
      name: "Drama3MLX",
      dependencies: [
        "Drama3Contracts", .product(name: "MLXAudioCodecs", package: "mlx-audio-swift"),
      ]),
    .executableTarget(name: "WeeToddDrama3Worker", dependencies: ["Drama3MLX", "Drama3Contracts"]),
    .testTarget(name: "Drama3ContractsTests", dependencies: ["Drama3Contracts"]),
    .testTarget(name: "Drama3MLXTests", dependencies: ["Drama3MLX"]),
  ], swiftLanguageModes: [.v5])
