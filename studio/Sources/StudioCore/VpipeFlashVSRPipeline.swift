import Foundation

/// Pure translation into a pinned external graph. No download or registry stages.
public enum VpipeFlashVSRPipeline {
  public static func make(settings: FlashVSRSettings, modelRoot: String, input: String,
    output: String, fps: Double, frames: Int) throws -> Data {
    try settings.validateExecution()
    guard settings.backend == .vpipe else { throw StudioError.invalid("A native FlashVSR request cannot be translated into a vpipe graph.") }
    guard fps.isFinite, (1...120).contains(fps), frames > 0,
      [modelRoot, input, output].allSatisfy({ $0.hasPrefix("/") && !$0.utf8.contains(0) }) else {
      throw StudioError.invalid("FlashVSR+ graph needs local absolute paths and a valid complete frame interval.")
    }
    let canvas = try settings.canvas()
    func port(_ source: String, _ index: Int = 0) -> [String: Any] { ["src": source, "oport": index] }
    func stage(_ id: String, _ type: String? = nil, _ inputs: [[String: Any]], _ config: [String: Any]) -> [String: Any] {
      ["id": id, "type": type ?? id, "iports": inputs, "config": config]
    }
    let stages: [[String: Any]] = [
      stage("model-select", nil, [], ["hf_dir": URL(fileURLWithPath: modelRoot).appendingPathComponent(settings.variant.rawValue).path]),
      stage("load-video", nil, [], ["input_url": input, "enable_audio": false]),
      stage("video-to-rgb", nil, [port("load-video")], ["output_dtype": "u8", "hwaccel": "auto", "oport_capacity": 2]),
      stage("visible-input", "image-resample", [port("video-to-rgb")], ["width": settings.width, "height": settings.height,
        "fit": settings.fit == .fit ? "pad" : "crop", "algorithm": "bicubic"]),
      stage("canvas-input", "image-resample", [port("visible-input")], ["width": canvas.width, "height": canvas.height,
        "fit": "manual", "src_x": 0, "src_y": 0, "scale": 1.0]),
      stage("temporal-stack", nil, [port("canvas-input")], ["mode": "video", "group_size": 24, "overlap": 0, "max_mb": 512]),
      stage("flashvsr-src-encoder", nil, [port("temporal-stack"), port("model-select")], ["decoder": settings.decoder.rawValue,
        "continuity": settings.continuity.rawValue, "color_fix": settings.colorFix != .off,
        "color_fix_method": settings.colorFix == .off ? "wavelet" : settings.colorFix.rawValue,
        "wavelet_levels": settings.waveletLevels, "temporal_compression": settings.temporalCompression,
        "temporal_merge": "gaussian", "temporal_sigma": settings.temporalSigma]),
      stage("generate-video", nil, [port("flashvsr-src-encoder"), port(""), port("model-select")], ["width": canvas.width,
        "height": canvas.height, "frames": frames, "fps": fps, "seed": settings.seed, "unload_when_idle": "auto"]),
      stage("vae-decode", nil, [port("generate-video"), port("model-select"), port("flashvsr-src-encoder", 1)], [:]),
      stage("visible-output", "image-resample", [port("vae-decode")], ["width": settings.width, "height": settings.height,
        "fit": "manual", "src_x": 0, "src_y": 0, "scale": 1.0]),
      stage("rgb-to-video", nil, [port("visible-output")], ["fps": fps, "pix_fmt": "rgb24", "colorspace": "bt709", "color_range": "full"]),
      stage("save-video", nil, [port("rgb-to-video")], ["output_url": output, "enable_video": true,
        "enable_audio": false, "video_codec": "rawvideo", "format": "nut", "no_overwrite": true])
    ]
    return try JSONSerialization.data(withJSONObject: ["id": "weetodd-flashvsr-turbo-v1", "stages": stages, "subpipelines": []], options: [.sortedKeys])
  }
}
