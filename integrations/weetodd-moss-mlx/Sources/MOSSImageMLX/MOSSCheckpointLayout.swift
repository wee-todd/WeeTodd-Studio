import Foundation
import Darwin
import TensorIO

public final class MOSSCheckpointIndex {
  public let directory: URL
  public let descriptors: [String: TensorDescriptor]
  let quantization: MOSSQuantizationManifest?
  private let manifestIdentity: String?
  let files: [URL: SafeTensorFile]
  let owners: [String: URL]
  let rawNames: [String: String]
  public init(directory: URL) throws {
    self.directory = directory.standardizedFileURL.resolvingSymlinksInPath()
    let directory = self.directory
    quantization = directory.pathExtension == "safetensors" ? nil : try MOSSQuantizationManifest.read(directory)
    manifestIdentity = quantization == nil ? nil : try Self.fileIdentity(directory.appendingPathComponent("moss_quantization.json"))
    let urls: [URL]
    if directory.pathExtension == "safetensors" { urls = [directory] }
    else { urls = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil).filter { $0.pathExtension == "safetensors" }.sorted { $0.path < $1.path } }
    guard !urls.isEmpty, urls.count <= 128 else { throw MOSSError("No bounded safetensors component found at \(directory.lastPathComponent).") }
    var files: [URL: SafeTensorFile] = [:], descriptors: [String: TensorDescriptor] = [:], owners: [String: URL] = [:]
    for url in urls {
      let file = try SafeTensorFile(url: url)
      for (name, tensor) in file.tensors {
        guard descriptors[name] == nil else { throw MOSSError("Duplicate tensor across component shards: \(name)") }
        descriptors[name] = tensor; owners[name] = url
      }
      files[url] = file
    }
    let prefixes = ["model.diffusion_model.", "diffusion_model.", "model.", "first_stage_model."]
    let markers = ["video_embedder.weight", "language_model.embed_tokens.weight", "decoder.conv_in.weight"]
    let prefix = prefixes.first { prefix in markers.contains { descriptors[prefix + $0] != nil } } ?? ""
    var normalized: [String: TensorDescriptor] = [:], normalizedOwners: [String: URL] = [:], rawNames: [String: String] = [:]
    for (raw, descriptor) in descriptors {
      let name = !prefix.isEmpty && raw.hasPrefix(prefix) ? String(raw.dropFirst(prefix.count)) : raw
      guard normalized[name] == nil else { throw MOSSError("Ambiguous tensor namespace: \(name)") }
      normalized[name] = descriptor; normalizedOwners[name] = owners[raw]; rawNames[name] = raw
    }
    self.files = files; self.descriptors = normalized; self.owners = normalizedOwners; self.rawNames = rawNames
  }
  public func require(_ name: String, shape: [Int], dtypes: Set<String> = ["BF16", "F32", "F16"]) throws {
    guard let descriptor = descriptors[name], descriptor.shape == shape.map(UInt64.init), dtypes.contains(descriptor.dtype) else {
      throw MOSSError("Missing or incompatible tensor \(name); expected \(shape).")
    }
  }
  public func checkUnchanged() throws {
    for (url, file) in files { try file.checkUnchanged(at: url) }
    if let manifestIdentity, try Self.fileIdentity(directory.appendingPathComponent("moss_quantization.json")) != manifestIdentity { throw MOSSError("Quantization metadata changed after admission.") }
  }
  private static func fileIdentity(_ url: URL) throws -> String {
    var s = stat()
    guard fstatat(AT_FDCWD, url.path, &s, 0) == 0 else { throw MOSSError("Checkpoint metadata is unavailable.") }
    return "\(url.path)|\(s.st_dev)|\(s.st_ino)|\(s.st_size)|\(s.st_mtimespec.tv_sec):\(s.st_mtimespec.tv_nsec)|\(s.st_ctimespec.tv_sec):\(s.st_ctimespec.tv_nsec)"
  }
  public func identity() throws -> String {
    try checkUnchanged()
    return try files.keys.sorted { $0.path < $1.path }.map { url in
      var s = stat()
      guard fstatat(AT_FDCWD, url.path, &s, 0) == 0 else { throw MOSSError("Checkpoint is unavailable.") }
      return "\(url.path)|\(s.st_dev)|\(s.st_ino)|\(s.st_size)|\(s.st_mtimespec.tv_sec):\(s.st_mtimespec.tv_nsec)|\(s.st_ctimespec.tv_sec):\(s.st_ctimespec.tv_nsec)"
    }.joined(separator: "\n") + (manifestIdentity ?? "")
  }
}
