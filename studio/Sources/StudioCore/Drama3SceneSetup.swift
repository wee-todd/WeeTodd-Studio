import Darwin
import Drama3Contracts
import Foundation

extension NativeModelDownloads {
  public static func dramaSceneSource(at catalog: URL) throws -> NativeModelDownloadFile {
    let sources = try self.catalog(at: catalog).filter { $0.kind == "drama3-native" }
      .flatMap(\.files).filter { $0.target == ".original-head-source.safetensors" }
    guard let source = sources.first,
      sources.allSatisfy({ $0.sha256 == source.sha256 && $0.size == source.size
        && $0.repo == source.repo && $0.revision == source.revision && $0.filename == source.filename })
    else { throw StudioError.invalid("The pinned Drama 3 scene-control source is unavailable.") }
    return source
  }

  /// Complete an installed model without reacquiring its generator or codec. Verified source
  /// bytes remain under that model until extraction succeeds; partial downloads survive retries.
  public static func prepareDramaSceneSource(catalog: URL, model: URL, existingRoots: [String],
                                             progress: @escaping Progress) async throws -> URL {
    try await prepareDramaSceneSource(catalog: catalog, model: model, existingRoots: existingRoots,
      progress: progress, transfer: { file, partial, token, progress in
        try await NativeModelHTTPTransfer(file: file, partial: partial, token: token, progress: progress).run()
      })
  }

  static func prepareDramaSceneSource(catalog: URL, model: URL, existingRoots: [String],
                                      progress: @escaping Progress, transfer: @escaping Transfer) async throws -> URL {
    let source = try dramaSceneSource(at: catalog)
    let root = model.standardizedFileURL.resolvingSymlinksInPath()
    var isDirectory: ObjCBool = false
    let fm = FileManager.default
    guard fm.fileExists(atPath: root.path, isDirectory: &isDirectory), isDirectory.boolValue else {
      throw StudioError.invalid("Reconnect the installed Drama 3 model folder before completing setup.")
    }
    let ownership = try Drama3SceneSetupLock(model: root)
    defer { withExtendedLifetime(ownership) {} }
    let final = root.appendingPathComponent(source.target)
    if try verified(final, file: source) { return final }
    guard !fm.fileExists(atPath: final.path) else {
      throw StudioError.invalid("The retained scene-control source has a different checksum. Move it aside before retrying.")
    }
    progress("Checking shared model storage for the original scene-control shard…", 0)
    let candidates = try existingFiles(roots: existingRoots, sizes: [source.size], includeHidden: true)
    for candidate in candidates[source.size, default: []] {
      if try verified(candidate, file: source) {
        try Task.checkCancellation()
        do { try fm.linkItem(at: candidate, to: final) }
        catch {
          guard (error as NSError).domain == NSPOSIXErrorDomain && (error as NSError).code == Int(EXDEV)
            || (error as NSError).underlyingPOSIXCode == Int(EXDEV) else { throw error }
          try fm.createSymbolicLink(at: final, withDestinationURL: candidate)
        }
        return final
      }
    }
    let partial = final.appendingPathExtension("partial")
    let used = (try? fm.attributesOfItem(atPath: partial.path)[.size] as? NSNumber)?.int64Value ?? 0
    let remaining = source.size - max(0, min(used, source.size))
    let free = try fm.attributesOfFileSystem(forPath: root.path)[.systemFreeSize] as? NSNumber
    guard let free, free.int64Value >= remaining + 256 * 1024 * 1024 else {
      throw StudioError.invalid("The model folder needs more free space for the original scene-control shard.")
    }
    try await transfer(source, partial, nil, progress)
    guard try verified(partial, file: source) else {
      if (try? fm.attributesOfItem(atPath: partial.path)[.size] as? NSNumber)?.int64Value == source.size {
        try? fm.removeItem(at: partial)
      }
      throw StudioError.invalid("Scene-control source checksum differs from its pinned release. Retry setup.")
    }
    try Task.checkCancellation()
    try fm.moveItem(at: partial, to: final)
    return final
  }
}
