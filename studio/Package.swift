// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "WeeToddStudio",
  platforms: [.macOS(.v14)],
  products: [
    .executable(name: "WeeToddStudio", targets: ["WeeToddStudio"]),
    .executable(name: "StudioMetal", targets: ["StudioMetal"]),
    .executable(name: "WeeToddCLI", targets: ["WeeToddCLI"]),
    .library(name: "StudioCore", targets: ["StudioCore"]),
  ],
  dependencies: [.package(path: "../integrations/weetodd-drama3"),
    .package(path: "../integrations/weetodd-inference")],
  targets: [
    .target(name: "StudioCore", dependencies: [.product(name: "Drama3Contracts", package: "weetodd-drama3"),
      .product(name: "InferenceContracts", package: "weetodd-inference"),
      .product(name: "TensorIO", package: "weetodd-inference")]),
    .executableTarget(name: "WeeToddStudio", dependencies: ["StudioCore"]),
    .executableTarget(name: "StudioMetal"),
    .executableTarget(name: "WeeToddCLI", dependencies: ["StudioCore"]),
    .testTarget(name: "StudioCoreTests", dependencies: ["StudioCore"]),
    .testTarget(name: "StudioAppTests", dependencies: ["WeeToddStudio"]),
  ],
  swiftLanguageModes: [.v5]
)
