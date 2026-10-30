import Foundation

public enum FlashVSRModelDownloader {
  public struct Requirement: Sendable {
    public let relativePath: String, url: String, sha256: String
    public let bytes: Int64
  }
  public static func requirements(decoder: FlashVSRSettings.Decoder) throws -> [Requirement] {
    let base = "https://huggingface.co/ruchangyao/FlashVSR_plus/resolve/46b79520946ac67dec94d1566490690953c7debc/"
    var files: [Requirement] = [
      .init(relativePath: "turbo/dit.safetensors", url: base + "turbo/dit.safetensors", sha256: "7862b92178176fa3aebc34c6c071daca6b56ab1dddb8e7ff9a01c47f5226ec8b", bytes: 5676070392),
      .init(relativePath: "turbo/lq_proj_in.safetensors", url: base + "turbo/lq_proj_in.safetensors", sha256: "e77b378ae12342eefbb58f924256f362a151b93ce73dd32808fcdaca9c051496", bytes: 1151384272),
      .init(relativePath: "posi_prompt.pth", url: "https://raw.githubusercontent.com/OpenImagingLab/FlashVSR_Plus/b548a2cc2c929d3d4676f2973abe27876ec4dee7/prompt_tensor/posi_prompt.pth", sha256: FlashVSRModelManifest.contextSHA256, bytes: 4195504),
      .init(relativePath: "config.json", url: base + "config.json", sha256: "13f6ebddb89f61223a06b9e3de1293647dbe74b639ccc45e01bd353a5b6e2577", bytes: 283)
    ]
    files.append(decoder == .tiny
      ? .init(relativePath: "TCDecoder.ckpt", url: base + "TCDecoder.ckpt", sha256: "e224bdcf2f52745cbf4d393ff5374c2ba09e90285d5d19062d2bf63b915b6161", bytes: 189018333)
      : .init(relativePath: "Wan2.1_VAE.pth", url: base + "Wan2.1_VAE.pth", sha256: "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981", bytes: 507609880))
    return files
  }
  public static func download(root: URL, decoder: FlashVSRSettings.Decoder, logDirectory: URL,
    emit: @escaping @Sendable (String) -> Void = { _ in }) async throws {
    let cancellation = NativeHeadlessCancellation()
    try await withTaskCancellationHandler(operation: {
      try await Task.detached {
        var isDirectory: ObjCBool = false
        guard root.isFileURL, FileManager.default.fileExists(atPath: root.path, isDirectory: &isDirectory), isDirectory.boolValue else {
          throw StudioError.invalid("Reconnect the model volume and select an existing FlashVSR+ folder before downloading.")
        }
        let required = try requirements(decoder: decoder)
        var missing: [Requirement] = []
        for item in required {
          let destination = root.appendingPathComponent(item.relativePath)
          if FileManager.default.fileExists(atPath: destination.path) {
            let file = try FlashVSRFile.capture(destination)
            guard file.sha256 == item.sha256, file.bytes == item.bytes else {
              throw StudioError.invalid("Existing " + item.relativePath + " differs from the official checkpoint. Relink the correct files or select an empty folder; existing weights are not overwritten.")
            }
          } else { missing.append(item) }
        }
        let space = (try FileManager.default.attributesOfFileSystem(forPath: root.path)[.systemFreeSize] as? NSNumber)?.int64Value ?? 0
        guard space > missing.reduce(Int64(0), { $0 + $1.bytes }) else { throw StudioError.invalid("The model volume does not have enough free space for the missing components.") }
        try FileManager.default.createDirectory(at: logDirectory, withIntermediateDirectories: true)
        for item in missing {
          try cancellation.check()
          let destination = root.appendingPathComponent(item.relativePath), partial = destination.appendingPathExtension("download")
          try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
          if let attrs = try? FileManager.default.attributesOfItem(atPath: partial.path), attrs[.type] as? FileAttributeType != .typeRegular {
            throw StudioError.invalid("The partial model download must be a regular file.")
          }
          emit("Downloading " + item.relativePath + " · " + String(item.bytes / 1_000_000) + " MB")
          let result = try ExternalEngineProcess.run(executable: URL(fileURLWithPath: "/usr/bin/curl"),
            arguments: ["--fail", "--location", "--retry", "3", "--continue-at", "-", "--silent", "--show-error", "--output", partial.path, item.url],
            directory: root, log: logDirectory.appendingPathComponent(UUID().uuidString + ".log"), cancellation: cancellation)
          guard result.exitCode == 0 else { throw StudioError.invalid("The model download stopped. Retry to resume the partial file.") }
          let downloaded = try FlashVSRFile.capture(partial)
          guard downloaded.sha256 == item.sha256, downloaded.bytes == item.bytes else {
            throw StudioError.invalid("Downloaded " + item.relativePath + " failed its official size or SHA-256 check. The partial file was not installed.")
          }
          try FileManager.default.moveItem(at: partial, to: destination)
        }
        try cancellation.check(); emit("Official Turbo model and selected decoder verified.")
      }.value
    }, onCancel: { cancellation.cancel() })
  }
}
