import CryptoKit
import Darwin
import Foundation

/// Weight-free Studio preparation for the shared Swift Ripple worker.
public enum NativeRipplePreparation {
  private static func regular(_ path: String, label: String) throws -> URL {
    guard path.hasPrefix("/"), !path.utf8.contains(0) else {
      throw StudioError.invalid("Select an absolute local \(label) path.")
    }
    let url = URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
    var status = stat()
    guard Darwin.lstat(url.path, &status) == 0, status.st_mode & S_IFMT == S_IFREG,
      status.st_size > 0 else { throw StudioError.invalid("The Ripple \(label) file is missing.") }
    return url
  }

  private static func digest(_ path: String) throws -> String {
    let url = try regular(path, label: "source movie")
    let fd = Darwin.open(url.path, O_RDONLY | O_CLOEXEC | O_NONBLOCK)
    guard fd >= 0 else { throw StudioError.invalid("Cannot read the Ripple source movie.") }
    let file = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
    defer { try? file.close() }
    var before = stat()
    guard fstat(fd, &before) == 0 else { throw StudioError.invalid("Cannot identify the Ripple source movie.") }
    var hash = SHA256()
    while let bytes = try file.read(upToCount: 8 * 1024 * 1024), !bytes.isEmpty {
      try Task.checkCancellation()
      hash.update(data: bytes)
    }
    var after = stat()
    guard fstat(fd, &after) == 0, before.st_size == after.st_size,
      before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec,
      before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec else {
      throw StudioError.invalid("The Ripple source movie changed during preparation.")
    }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
  }

  private static func executable(_ runtime: [String: Any]) throws -> String {
    let configured = (runtime["ffmpegPath"] as? String ?? "").trimmingCharacters(in: .whitespaces)
    if !configured.isEmpty {
      guard configured.hasPrefix("/"), FileManager.default.isExecutableFile(atPath: configured) else {
        throw StudioError.invalid("Select an executable FFmpeg in Runtime Settings.")
      }
      return configured
    }
    for directory in (ProcessInfo.processInfo.environment["PATH"] ?? "").split(separator: ":") {
      let candidate = String(directory) + "/ffmpeg"
      if FileManager.default.isExecutableFile(atPath: candidate) { return candidate }
    }
    throw StudioError.invalid("Install FFmpeg or select it in Runtime Settings.")
  }

  private static func profile(_ runtime: [String: Any]) throws -> [String: String] {
    let selected = (runtime["rippleProfileID"] as? String ?? "").trimmingCharacters(in: .whitespaces)
    let folder = runtime["profilesDirectory"] as? String ?? ""
    let candidates: [URL]
    if !selected.isEmpty && selected != "auto" {
      guard selected.hasPrefix("/") else {
        throw StudioError.invalid("Select an absolute installed LTX 2.5 profile path.")
      }
      candidates = [URL(fileURLWithPath: selected)]
    } else {
      guard folder.hasPrefix("/"),
        let entries = try? FileManager.default.contentsOfDirectory(at: URL(fileURLWithPath: folder),
          includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]),
        entries.count <= 1000 else {
        throw StudioError.invalid("Select a model profiles folder with at most 1,000 files.")
      }
      candidates = entries.filter { $0.pathExtension == "json" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }
    for candidate in candidates {
      try Task.checkCancellation()
      guard let metadata = try? candidate.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
        metadata.isRegularFile == true, let size = metadata.fileSize,
        size <= 1024 * 1024,
        let bytes = try? Data(contentsOf: candidate, options: .mappedIfSafe),
        bytes.count <= 1024 * 1024,
        let value = try? JSONSerialization.jsonObject(with: bytes) as? [String: Any],
        value["format"] as? String == "weetodd-headless-v2",
        value["engine"] as? String == "ltx25",
        let config = value["config"] as? [String: Any],
        (config["pipeline_mode"] as? String ?? "distilled") == "distilled",
        let components = value["components"] as? [String: Any],
        (components["loras"] as? [Any] ?? []).isEmpty,
        (components["ic_loras"] as? [Any] ?? []).isEmpty,
        ["msr_lora_path", "distilled_lora_path"].allSatisfy({
          (components[$0] as? String ?? "").isEmpty
        }) else { continue }
      var resolved: [String: String] = [:]
      var valid = true
      for key in ["transformer_path", "text_encoder_path", "video_vae_path", "audio_vae_path"] {
        guard let path = components[key] as? String, !path.isEmpty else { valid = false; break }
        let expanded = (path as NSString).expandingTildeInPath
        guard expanded.hasPrefix("/"), FileManager.default.fileExists(atPath: expanded) else {
          valid = false; break
        }
        resolved[key] = URL(fileURLWithPath: expanded).standardizedFileURL.resolvingSymlinksInPath().path
      }
      if valid { return resolved }
    }
    throw StudioError.invalid("Select a plain installed LTX 2.5 distilled profile for Ripple.")
  }

  public static func prepare(draft: RippleDraft, runtime: [String: Any],
    into directory: URL, output: URL) async throws -> [String: Any] {
    try draft.validate()
    guard draft.frameCount <= 1501, draft.references.count <= 9,
      !FileManager.default.fileExists(atPath: output.path) else {
      throw StudioError.invalid("Ripple exceeds the native frame window or its output already exists.")
    }
    let adapter = try regular(runtime["rippleAdapterPath"] as? String ?? "",
      label: "author adapter")
    let components = try profile(runtime)
    let ffmpeg = try executable(runtime)
    let raw = try draft.bridgeObject()
    let inspection = try await NativeRippleMedia.inspect(raw)
    guard inspection["frame_count"] as? Int == draft.frameCount,
      inspection["width"] as? Int == draft.width,
      inspection["height"] as? Int == draft.height else {
      throw StudioError.invalid("Ripple inspection differs from the current draft.")
    }
    let sourceHash = try digest(draft.sourcePath)
    if let expected = draft.sourceSHA256, expected != sourceHash {
      throw StudioError.invalid("The Ripple source movie differs from the saved take.")
    }
    let fm = FileManager.default
    try fm.createDirectory(at: directory, withIntermediateDirectories: true)
    var finished = false
    defer { if !finished { try? fm.removeItem(at: directory) } }
    let sorted = draft.references.sorted { $0.frame < $1.frame }
    var frozen: [(frame: Int, path: String, strength: Double)] = []
    for (index, reference) in sorted.enumerated() {
      let source = try regular(reference.path, label: "edited reference")
      let attributes = try fm.attributesOfItem(atPath: source.path)
      guard (attributes[.size] as? NSNumber)?.int64Value ?? 0 <= 64 * 1024 * 1024 else {
        throw StudioError.invalid("Ripple reference images must be at most 64 MiB each.")
      }
      let target = directory.appendingPathComponent(String(format: "edited-%02d-frame-%06d", index + 1,
        reference.frame) + "." + source.pathExtension.lowercased())
      try fm.copyItem(at: source, to: target)
      frozen.append((reference.frame, target.path, reference.strength))
    }
    guard let first = frozen.first, first.frame == 0 else {
      throw StudioError.invalid("Ripple needs an edited first frame.")
    }
    let guide = try await NativeRippleMedia.prepareGuide(raw,
      editedFirstFrame: URL(fileURLWithPath: first.path),
      into: directory.appendingPathComponent("guide"))
    guard let modelFrames = guide["frames"] as? Int,
      let guidePath = guide["rgb_path"] as? String,
      modelFrames <= 1501 else { throw StudioError.invalid("Ripple guide exceeds the native window.") }
    guard try digest(draft.sourcePath) == sourceHash else {
      throw StudioError.invalid("The Ripple source movie changed during guide preparation.")
    }
    let transformer = components["transformer_path"]!
    let prompt = draft.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    let request: [String: Any] = [
      "version": 1, "engine": "ltx25", "task": "ripple",
      "gemma_root": components["text_encoder_path"]!, "transformer_root": transformer,
      "connector_checkpoint": transformer + "/pages/fixed.safetensors",
      "video_checkpoint": components["video_vae_path"]!,
      "audio_checkpoint": components["audio_vae_path"]!,
      "adapter_path": adapter.path, "adapter_strength": draft.loraStrength,
      "guide_path": guidePath, "first_reference_path": first.path,
      "source_path": draft.sourcePath, "source_sha256": sourceHash,
      "source_start": draft.sourceIn, "duration": draft.duration,
      "editorial_frames": draft.frameCount,
      "width": draft.width, "height": draft.height, "frames": modelFrames,
      "fps": draft.frameRate, "seed": draft.seed,
      "prompt": prompt.isEmpty ? RippleDraft.defaultPrompt : prompt,
      "reference_strength": first.strength,
      "anchors": frozen.dropFirst().map {
        ["frame": $0.frame, "path": $0.path, "strength": $0.strength] as [String: Any]
      },
      "audio_policy": draft.audioPolicy.rawValue,
      "ffmpeg_path": ffmpeg, "output_directory": output.path]
    let recipe = directory.appendingPathComponent("ripple-request.json")
    try JSONSerialization.data(withJSONObject: request, options: [.prettyPrinted, .sortedKeys])
      .write(to: recipe, options: .withoutOverwriting)
    finished = true
    return ["recipePath": recipe.path, "modelFrames": modelFrames,
      "editorialFrames": draft.frameCount, "sourceSHA256": sourceHash]
  }
}
