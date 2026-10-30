import CryptoKit
import Foundation
import InferenceContracts
import TensorIO

public struct NativeUpscalerConversionReceipt: Codable, Equatable, Sendable {
  public let version: Int
  public let sourceSHA256: String
  public let outputSHA256: String
  public let outputPath: String
  public let profile: String
  public let importerVersion: String
  public let conversionKey: String
}

/// Setup is shared by Studio and the CLI. Archive imports are deliberately
/// restricted to the three released tensor layouts, with no executable loader.
public enum NativeUpscalerSetup {
  public static var capabilities: [NativeCapabilityDescriptor] {
    [.h3Regenerate] + NativeCapabilityDescriptor.flashProfiles
  }
  public static let sharpTableSHA256 = "c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7"

  public static func download(packageID: String, catalog: URL, modelLibrary: URL,
    reuseRoots: [String] = [], token: String? = nil, progress: @escaping NativeModelDownloads.Progress = { _, _ in }) async throws -> URL {
    let library = try existingLibrary(modelLibrary)
    guard let package = try NativeModelDownloads.catalog(at: catalog).first(where: { $0.descriptor.id == packageID }),
      ["h3-regenerate", "h3-fast-vae", "h3-encoder-source", "h3-ref2va-source", "flashvsr-native"].contains(package.kind) else {
      throw StudioError.invalid("Choose a pinned native H3 regeneration or FlashVSR+ package.")
    }
    let family = package.kind == "flashvsr-native" ? "FlashVSR-Plus"
      : package.kind == "h3-ref2va-source" ? "WeeTodd/H3/Ref2VA"
      : ["h3-fast-vae", "h3-encoder-source"].contains(package.kind) ? "WeeTodd/H3-Video-VAE" : "WeeTodd/LoRAs/H3"
    let destination = library.appendingPathComponent(family)
    return try await NativeModelDownloads.prepare(id: packageID, catalog: catalog,
      destination: destination, existingRoots: [library.path, library.appendingPathComponent("WeeTodd/H3-Upscalers").path, destination.path] + reuseRoots, token: token, progress: progress)
  }

  public static func convert(profile name: String, source: URL,
    modelLibrary: URL) throws -> NativeUpscalerConversionReceipt {
    let profile: TorchTensorImportProfile
    switch name {
    case "tiny": profile = .tinyDecoder
    case "wan": profile = .wanVAE
    case "prompt": profile = .positivePrompt
    default: throw StudioError.invalid("Choose the released tiny, wan or prompt archive profile.")
    }
    let archive = try TorchTensorArchive.open(url: source.resolvingSymlinksInPath(),
      expectedSHA256: profile.sourceSHA256, profile: profile)
    let library = try existingLibrary(modelLibrary)
    let directory = library.appendingPathComponent("FlashVSR-Plus/swift-mlx/" + archive.conversionKey)
    let output = directory.appendingPathComponent(name + ".safetensors")
    let receiptURL = directory.appendingPathComponent("conversion.json")
    if FileManager.default.fileExists(atPath: output.path) {
      let data = try boundedData(receiptURL, count: 16 * 1024)
      let receipt = try JSONDecoder().decode(NativeUpscalerConversionReceipt.self, from: data)
      guard receipt.version == 1, receipt.sourceSHA256 == archive.sourceSHA256,
        receipt.profile == profile.id, receipt.importerVersion == TorchTensorArchive.importerVersion,
        receipt.conversionKey == archive.conversionKey, receipt.outputPath == output.path,
        receipt.outputSHA256 == (try NativeMovieFrozenMedia.digest(output)) else {
        throw StudioError.invalid("The converted tensor cache changed. Choose another library or remove that cache after checking it.")
      }
      return receipt
    }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let hash = try archive.importSafetensors(to: output)
    let receipt = NativeUpscalerConversionReceipt(version: 1, sourceSHA256: archive.sourceSHA256,
      outputSHA256: hash, outputPath: output.path, profile: profile.id,
      importerVersion: TorchTensorArchive.importerVersion, conversionKey: archive.conversionKey)
    do {
      try Task.checkCancellation()
      try JSONEncoder().encode(receipt).write(to: receiptURL, options: .withoutOverwriting)
    } catch {
      try? FileManager.default.removeItem(at: output)
      throw error
    }
    return receipt
  }

  /// The 8-dimensional Sharp coordinates cannot be reconstructed from the
  /// native base's different centered 64-dimensional table.
  public static func prepareSharpTable(modelLibrary: URL, existingSources: [URL] = []) async throws -> URL {
    let library = try existingLibrary(modelLibrary)
    let destination = library.appendingPathComponent("WeeTodd/LoRAs/H3/Open-H3-Regenerate-2K/adaln_t_table.bin")
    if FileManager.default.fileExists(atPath: destination.path) {
      guard validSharp(try boundedData(destination, count: 32800)) else { throw StudioError.invalid("Existing Sharp timestep coordinates differ from the pinned publisher table.") }
      return destination
    }
    var payload: Data?
    for source in existingSources {
      try Task.checkCancellation()
      guard let file = try? SafeTensorFile(url: source), let tensor = file.tensors["adaln_t_table"],
        tensor.shape == [1025, 8], tensor.dtype == "F32", tensor.byteCount == 32800 else { continue }
      let bytes = try file.withTensorBytes(named: "adaln_t_table") { Data($0) }
      try file.checkUnchanged(at: source)
      if validSharp(bytes) { payload = bytes; break }
    }
    if payload == nil {
      let source = URL(string: "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/e5eb578a89295337b8ff433a035929ce0279e0b6/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors")!
      let prefix = try await range(source, start: 0, count: 8)
      let length = prefix.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
      guard (2...8 * 1024 * 1024).contains(length) else { throw StudioError.invalid("Invalid source tensor header size.") }
      let bytes = try await range(source, start: 8, count: Int(length))
      guard let header = try JSONSerialization.jsonObject(with: bytes) as? [String: Any],
        let table = header["adaln_t_table"] as? [String: Any], table["dtype"] as? String == "F32",
        table["shape"] as? [Int] == [1025, 8], let offsets = table["data_offsets"] as? [UInt64],
        offsets.count == 2, offsets[1] >= offsets[0], offsets[1] - offsets[0] == 32800,
        offsets[1] <= UInt64(Int64.max) - length - 8 else {
        throw StudioError.invalid("The pinned source does not expose the required Sharp coordinates.")
      }
      payload = try await range(source, start: length + 8 + offsets[0], count: 32800)
    }
    guard let payload, validSharp(payload) else { throw StudioError.invalid("Sharp coordinate checksum differs from the publisher's table.") }
    try Task.checkCancellation()
    try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
    try payload.write(to: destination, options: .withoutOverwriting)
    return destination
  }

  private static func existingLibrary(_ url: URL) throws -> URL {
    let root = url.standardizedFileURL.resolvingSymlinksInPath()
    let attributes = try root.resourceValues(forKeys: [.isDirectoryKey])
    guard root.isFileURL, attributes.isDirectory == true else { throw StudioError.invalid("Reconnect and select an existing model library before setup.") }
    return root
  }
  private static func validSharp(_ data: Data) -> Bool {
    data.count == 32800 && SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == sharpTableSHA256
  }
  private static func boundedData(_ url: URL, count: Int) throws -> Data {
    let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
    guard values.isRegularFile == true, let size = values.fileSize, size > 0, size <= count else {
      throw StudioError.invalid("Invalid bounded native setup metadata.")
    }
    let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
    let data = try handle.read(upToCount: count + 1) ?? Data()
    guard data.count == size else { throw StudioError.invalid("Native setup metadata changed while reading.") }
    return data
  }
  private static func range(_ url: URL, start: UInt64, count: Int) async throws -> Data {
    try Task.checkCancellation()
    var request = URLRequest(url: url)
    request.setValue("bytes=\(start)-\(start + UInt64(count) - 1)", forHTTPHeaderField: "Range")
    let (bytes, response) = try await URLSession.shared.bytes(for: request)
    guard let http = response as? HTTPURLResponse, http.statusCode == 206,
      http.value(forHTTPHeaderField: "Content-Range")?.hasPrefix("bytes \(start)-\(start + UInt64(count) - 1)/") == true else {
      throw StudioError.invalid("The source did not honor the bounded Sharp table range request.")
    }
    var result = Data(); result.reserveCapacity(count)
    for try await byte in bytes {
      guard result.count < count else { throw StudioError.invalid("The source exceeded its bounded range.") }
      result.append(byte)
      if result.count % 4096 == 0 { try Task.checkCancellation() }
    }
    guard result.count == count else { throw StudioError.invalid("The Sharp table transfer was incomplete.") }
    return result
  }
}
