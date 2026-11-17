import Foundation

public enum ReCaReleasedAssets {
  public struct Pin: Sendable {
    public let role: ReCaAssetRole, relativePath: String, sha256: String
    public let bytes: UInt64
  }
  public static func pins(_ precision: ReCaPrecision) -> [Pin] {
    [.init(role: .transformer, relativePath: precision == .bf16 ? "official/transformer.safetensors" : "int8/transformer_int8_convrot.safetensors",
      sha256: precision == .bf16 ? "a43bc91689de37d286bfb1068e90049b2e121bbabb1cf4e6f5e9a801107e23ae" : "70f77ac653a29dcab8cb8d5f5300acf27d9a3a45e9baf273fa4526fa43f66c9d",
      bytes: precision == .bf16 ? 10_603_673_624 : 6_266_935_048),
     .init(role: .prompt, relativePath: "shared/prompt.safetensors", sha256: "325ee3b5ec7d8bd43bc7ab186be5272dec0877c04e7c01a46fd0893260c9c6a5", bytes: 4_194_392),
     .init(role: .flashDecoder, relativePath: "shared/flashdecoder.safetensors", sha256: "f4feb3e7b0b65f98e15bb709a1ba28dfaada7c06bf7fc0ce4cdbcb58803034e8", bytes: 228_723_776),
     .init(role: .modelConfig, relativePath: "shared/model_config.json", sha256: "0b4bd69d5b05304861559cdcd079a3b0e6f72c295ab88d16293d612900d9c9bf", bytes: 1_145),
     .init(role: .vaeConfig, relativePath: "shared/vae_config.json", sha256: "c50287da17bf094fbd67c328970725c43913ec7467affc055621cf235de48ee2", bytes: 1_599),
     .init(role: .flashDecoderConfig, relativePath: "shared/flashdecoder_config.json", sha256: "d89abe4c6b223d9a71fb975ac0037ef0bd792515f666a17446d8b92f9f42068c", bytes: 88)]
  }
  /// Resolve installed files without loading or hashing weights on the UI thread.
  /// Full verification remains mandatory before inference.
  public static func resolve(directory: URL, precision: ReCaPrecision) throws -> ReCaAssets {
    let files = try pins(precision).map { pin -> ReCaAsset in
      let candidates = [directory.appendingPathComponent(pin.relativePath),
        directory.appendingPathComponent(URL(fileURLWithPath: pin.relativePath).lastPathComponent)]
      guard let url = candidates.first(where: { FileManager.default.isReadableFile(atPath: $0.path) }) else {
        throw ContractError.invalid("Missing ReCaVSR file: \(URL(fileURLWithPath: pin.relativePath).lastPathComponent)")
      }
      let canonical = url.resolvingSymlinksInPath()
      let values = try canonical.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
      guard values.isRegularFile == true, values.fileSize.map(UInt64.init) == pin.bytes else {
        throw ContractError.invalid("ReCaVSR asset size differs from the selected release: \(url.lastPathComponent)")
      }
      return ReCaAsset(role: pin.role, file: ReCaFileIdentity(path: canonical.path, sha256: pin.sha256, bytes: pin.bytes))
    }
    return try ReCaAssets(files: files)
  }
  public static func validate(_ assets: ReCaAssets, precision: ReCaPrecision) throws {
    try assets.validate()
    for pin in pins(precision) {
      let actual = assets[pin.role]
      guard actual.sha256 == pin.sha256, actual.bytes == pin.bytes else { throw ContractError.invalid("ReCaVSR requires the paired released checkpoint and configurations.") }
    }
  }
}
