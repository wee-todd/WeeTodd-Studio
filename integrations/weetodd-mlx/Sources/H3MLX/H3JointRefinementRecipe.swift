import CoreFoundation
import Foundation

/// Strict worker-level opt-in. The stripped ordinary recipe still passes its
/// original task validator before any artifact or media is read.
public enum H3JointRefinementRecipe {
  public struct Prepared {
    public let version: Int
    public let learnedUpscaler: URL?
    public let learnedUpscalerHeaderSHA256: String?
    public let ordinaryRecipe: Data
    public let saveFullLatents: Bool
    public let sourceManifest: URL?
    public let sourceSHA256: String?
    public let controls: H3JointRefinement?
    public let resizeMethod: H3SpatialLatentResizeMethod?
    public let targetGeometry: H3Geometry
    public let mode: String?
  }
  private static func number(_ value: Any?) throws -> Double {
    guard let n = value as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(), n.doubleValue.isFinite else {
      throw H3CheckpointError.invalid("H3 refinement needs finite numeric settings.")
    }
    return n.doubleValue
  }
  private static func integer(_ value: Any?) throws -> Int {
    let value = try number(value)
    guard value.rounded() == value, value >= 0, value < Double(Int.max) else {
      throw H3CheckpointError.invalid("H3 refinement needs bounded integer settings.")
    }
    return Int(value)
  }
  private static func boolean(_ value: Any?, default fallback: Bool) throws -> Bool {
    guard let value else { return fallback }
    guard let n = value as? NSNumber, CFGetTypeID(n) == CFBooleanGetTypeID() else {
      throw H3CheckpointError.invalid("H3 refinement requires explicit Boolean controls.")
    }
    return n.boolValue
  }
  public static func prepare(data: Data) throws -> Prepared? {
    guard data.count <= 1_048_576,
      var root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw H3CheckpointError.invalid("Invalid bounded H3 refinement recipe.")
    }
    guard root["joint_latents"] != nil || root["refinement"] != nil else { return nil }
    guard root["continuation"] == nil,
      let config = root["config"] as? [String: Any] else {
      throw H3CheckpointError.invalid("H3 initialized refinement cannot combine saved-tail continuation.")
    }
    let rawRefinement = root["refinement"] as? [String: Any]
    let version = try rawRefinement.map { try integer($0["version"]) } ?? 1
    guard [1,2].contains(version), version == 1 || rawRefinement?["mode"] as? String == "spatial" else {
      throw H3CheckpointError.invalid("Only explicit spatial refinement v2 admits the expanded H3 target.")
    }
    let canvasAdmission: H3CanvasAdmission = version == 2 ? .spatialRefinement : .ordinary
    let geometry = try H3Geometry(width: integer(config["width"]), height: integer(config["height"]),
      durationSeconds: number(config["duration_seconds"]), canvasAdmission: canvasAdmission)
    guard geometry.width * geometry.height <= canvasAdmission.maximumPixels else {
      throw H3CheckpointError.invalid("H3 refinement target exceeds the admitted canvas budget.")
    }
    try canvasAdmission.validatePackedRows(geometry.videoRows + geometry.audioRows + 1)
    var learnedURL: URL?, learnedHash: String?
    var save = false
    if let value = root.removeValue(forKey: "joint_latents") {
      guard let fields = value as? [String: Any], Set(fields.keys) == ["version", "save_full"],
        try integer(fields["version"]) == 1 else {
        throw H3CheckpointError.invalid("H3 joint_latents v1 requires an explicit save_full control.")
      }
      save = try boolean(fields["save_full"], default: false)
      guard save else { throw H3CheckpointError.invalid("Remove joint_latents when full latent publication is disabled.") }
    }
    var manifest: URL?, hash: String?, controls: H3JointRefinement?, method: H3SpatialLatentResizeMethod?, mode: String?
    if let value = root.removeValue(forKey: "refinement") {
      guard let fields = value as? [String: Any],
        Set(fields.keys).isSubset(of: ["version", "mode", "source_manifest", "source_manifest_sha256", "strength", "start_video_sigma", "evaluations", "preserve_audio", "resize_method", "learned_upscaler_path", "learned_upscaler_header_sha256"]),
        try integer(fields["version"]) == version,
        let selected = fields["mode"] as? String, ["initialized", "spatial", "motion"].contains(selected),
        let path = fields["source_manifest"] as? String, path.hasPrefix("/"), !path.utf8.contains(0), !path.contains("://"),
        let digest = fields["source_manifest_sha256"] as? String,
        digest.utf8.count == 64, digest.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
        (selected == "spatial") == (fields["resize_method"] != nil || fields["learned_upscaler_path"] != nil),
        version == 2 || (fields["learned_upscaler_path"] == nil && fields["learned_upscaler_header_sha256"] == nil),
        selected == "spatial" || fields["learned_upscaler_header_sha256"] == nil,
        selected != "motion" || fields["start_video_sigma"] != nil else {
        throw H3CheckpointError.invalid("H3 refinement v1 needs an exact full AV source and compatible spatial/noise controls.")
      }
      if selected == "motion" {
        let components = root["components"] as? [String: Any]
        let conditioning = root["conditioning"] as? [String: Any]
        guard components?["task"] as? String == "t2va", components?["fun_controlnet"] == nil,
          conditioning?["task"] as? String == "t2v", (conditioning?["inputs"] as? [Any])?.isEmpty == true,
          try integer(config["steps"]) >= 16,
          try number(fields["start_video_sigma"]) == number(fields["strength"]),
          try boolean(fields["preserve_audio"], default: true) else {
          throw H3CheckpointError.invalid("Motion repair needs plain full-schedule H3, matching explicit noise strength and preserved source audio.")
        }
        if let adapters = (root["loras"] as? [String: Any])?["adapters"] as? [[String: Any]] {
          for adapter in adapters {
            guard try adapter["start_after_evaluations"].map({ try integer($0) }) ?? 0 == 0 else {
              throw H3CheckpointError.invalid("Motion repair requires ordinary adapters active for its complete schedule.")
            }
          }
        }
      }
      mode = selected; manifest = URL(fileURLWithPath: path); hash = digest
      if selected == "spatial" {
        if fields["learned_upscaler_path"] != nil || fields["learned_upscaler_header_sha256"] != nil {
          guard version == 2, fields["resize_method"] == nil,
            let path = fields["learned_upscaler_path"] as? String, path.hasPrefix("/"),
            !path.utf8.contains(0), !path.contains("://"),
            let digest = fields["learned_upscaler_header_sha256"] as? String,
            digest.utf8.count == 64,
            digest.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw H3CheckpointError.invalid("Learned spatial refinement needs an exclusive path/header digest pair.")
          }
          learnedURL = URL(fileURLWithPath: path); learnedHash = digest
        } else {
        guard let text = fields["resize_method"] as? String, let value = H3SpatialLatentResizeMethod(rawValue: text) else {
          throw H3CheckpointError.invalid("Unsupported H3 spatial latent interpolation.")
        }
        method = value
        }
        guard fields["start_video_sigma"] == nil, fields["evaluations"] == nil else {
          throw H3CheckpointError.invalid("Spatial HiRes keeps the owned schedule-suffix semantics.")
        }
      }
      controls = try H3JointRefinement(strength: number(fields["strength"]),
        startVideoSigma: fields["start_video_sigma"].map { try number($0) },
        evaluations: fields["evaluations"].map { try integer($0) },
        preserveAudio: boolean(fields["preserve_audio"], default: true))
    }
    return Prepared(version: version, learnedUpscaler: learnedURL, learnedUpscalerHeaderSHA256: learnedHash, ordinaryRecipe: try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys]),
      saveFullLatents: save, sourceManifest: manifest, sourceSHA256: hash, controls: controls,
      resizeMethod: method, targetGeometry: geometry, mode: mode)
  }

  public static func validateSource(_ source: H3JointLatentArtifact.Manifest,
    prepared: Prepared) throws {
    let geometry = try source.geometry, target = prepared.targetGeometry
    if let url = prepared.learnedUpscaler, let digest = prepared.learnedUpscalerHeaderSHA256 {
      _ = try H3LearnedUpscalerLayout(url: url, expectedHeaderSHA256: digest)
      let plan = try H3UpscalerConvolutionPlan()
      _ = try plan.validate(channels: 512, kernel: 3, bytesPerElement: 4)
      _ = try plan.tileCount(frames: target.videoLatentFrames, height: target.height/16, width: target.width/16)
    }
    guard geometry.frames == target.frames,
      prepared.mode == "spatial" || (geometry.width == target.width && geometry.height == target.height),
      prepared.mode != "spatial" || (target.width > geometry.width && target.height > geometry.height &&
        target.width <= 2 * geometry.width && target.height <= 2 * geometry.height) else {
      throw H3CheckpointError.invalid("H3 full source timing/canvas differs from the explicit refinement mode.")
    }
  }
}
