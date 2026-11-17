// swift-tools-version: 6.0
import PackageDescription

let package = Package(
  name: "WeeToddImageLayout", platforms: [.macOS(.v14)],
  products: [.library(name: "ImageLayout", targets: ["ImageLayout"])],
  targets: [.target(name: "ImageLayout"), .testTarget(name: "ImageLayoutTests", dependencies: ["ImageLayout"])],
  swiftLanguageModes: [.v6])
