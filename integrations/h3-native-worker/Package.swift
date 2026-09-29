// swift-tools-version: 6.0
import PackageDescription
let package = Package(
  name: "WeeToddH3NativeWorker", platforms: [.macOS(.v14)],
  products: [.executable(name: "WeeToddH3Worker", targets: ["WeeToddH3Worker"])],
  dependencies: [.package(url: "https://github.com/liuliu/s4nnc.git",
    revision: "c99e45ad0d16c902c6c477e4dc9f1a7e49386dcd")],
  targets: [.executableTarget(name: "WeeToddH3Worker", dependencies: [
    .product(name: "NNC", package: "s4nnc")])],
  swiftLanguageModes: [.v5]
)
