import Foundation
import CoreFoundation
import LTX25Engine

/// Strict native scene admission. Each shot is compiled through the ordinary
/// Studio recipe validator after resolving its exact causal sampling window.
/// Each window may use one shot-boundary image alongside sampled audiovisual
/// history. One continuous audio driver can span every shot. Interior/last
/// images and unrelated per-shot audio sources remain gated.
public enum MLXStudioSceneRecipe {
  public struct Compiled {
    public let plan: LTX25ScenePlan
    public let requests: [MLXDistilledRequest]
    public let clipIDs: [String]
    public let decodeMode: MLXSceneDecodeMode
    public let strictBoundaries: Set<Int>
  }

  public static func compile(data: Data, outputDirectory: String) throws -> Compiled {
    guard data.count <= 1024 * 1024,
      let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      Set(root.keys).isSubset(of: ["format","engine","prompt","config","components",
        "conditioning","scene","ffmpeg","ffprobe","candidate"]),
      let scene = root["scene"] as? [String: Any],
      Set(scene.keys).isSubset(of: ["version","segments","overlap_frames",
        "boundary_image_policy","soundscape","music","decode_mode","decode_window_frames"]),
      let version = scene["version"] as? NSNumber,
      CFGetTypeID(version) != CFBooleanGetTypeID(), version.doubleValue == 1,
      let overlap = scene["overlap_frames"] as? NSNumber,
      CFGetTypeID(overlap) != CFBooleanGetTypeID(),
      overlap.doubleValue.isFinite, (9...4097).contains(overlap.doubleValue),
      overlap.doubleValue.rounded() == overlap.doubleValue,
      let policy = scene["boundary_image_policy"] as? String,
      ["balanced","strict"].contains(policy),
      (scene["soundscape"] as? String ?? "").isEmpty,
      (scene["music"] as? String ?? "").isEmpty,
      let entries = scene["segments"] as? [[String: Any]],
      (2...6).contains(entries.count),
      let config = root["config"] as? [String: Any],
      let fpsNumber = config["frame_rate"] as? NSNumber,
      CFGetTypeID(fpsNumber) != CFBooleanGetTypeID(),
      fpsNumber.doubleValue.isFinite, (1...120).contains(fpsNumber.doubleValue),
      let durationNumber = config["duration_seconds"] as? NSNumber,
      CFGetTypeID(durationNumber) != CFBooleanGetTypeID(),
      let conditioning = root["conditioning"] as? [String: Any],
      let sceneTask = conditioning["task"] as? String,
      let sceneInputs = conditioning["inputs"] as? [[String: Any]],
      (sceneTask == "t2v" && sceneInputs.isEmpty ||
        sceneTask == "fflf" && sceneInputs.count == 1 &&
        sceneInputs[0]["kind"] as? String == "image" &&
        sceneInputs[0]["role"] as? String == "keyframe" &&
        sceneInputs[0]["frame_index"] as? Int == 0 ||
        sceneTask == "a2v" && (1...2).contains(sceneInputs.count) &&
        sceneInputs.filter({ $0["role"] as? String == "audio_driver" &&
          $0["kind"] as? String == "audio" }).count == 1 &&
        sceneInputs.filter({ $0["role"] as? String == "keyframe" &&
          $0["kind"] as? String == "image" &&
          $0["frame_index"] as? Int == 0 }).count == sceneInputs.count - 1) else {
      throw LTXError.invalid("Swift LTX scenes need text, one opening image, or one continuous audio driver with an optional image.")
    }
    guard scene["decode_mode"] == nil || scene["decode_mode"] is String else {
      throw LTXError.invalid("Scene decode mode must be single or windowed.")
    }
    let decodeMode: MLXSceneDecodeMode
    switch scene["decode_mode"] as? String ?? "single" {
    case "single":
      guard scene["decode_window_frames"] == nil else {
        throw LTXError.invalid("A single-decode scene cannot set a decode window size.")
      }
      decodeMode = .single
    case "windowed":
      if let number = scene["decode_window_frames"] as? NSNumber {
        guard CFGetTypeID(number) != CFBooleanGetTypeID(),
          number.doubleValue.isFinite,
          number.doubleValue.rounded() == number.doubleValue,
          (33...4097).contains(number.intValue),
          (number.intValue - 1) % 8 == 0 else {
          throw LTXError.invalid("Scene decode windows must have an aligned 8n+1 frame count of at least 33.")
        }
        decodeMode = .windowed(maximumFrames: number.intValue)
      } else {
        guard scene["decode_window_frames"] == nil else {
          throw LTXError.invalid("Scene decode window size must be an integer.")
        }
        decodeMode = .windowed(maximumFrames: nil)
      }
    default:
      throw LTXError.invalid("Unsupported LTX scene video decode mode.")
    }
    var ids: [String] = [], prompts: [String] = [], durations: [Double] = []
    var seeds: [Int] = []
    for entry in entries {
      guard Set(entry.keys).subtracting(["image_input"]) == ["clip_id","prompt","duration_seconds","seed"],
        let id = entry["clip_id"] as? String, !id.isEmpty, id.utf8.count <= 128,
        let prompt = entry["prompt"] as? String,
        !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
        prompt.utf8.count <= 65_536,
        let duration = entry["duration_seconds"] as? NSNumber,
        CFGetTypeID(duration) != CFBooleanGetTypeID(),
        duration.doubleValue.isFinite, duration.doubleValue > 0,
        let seed = entry["seed"] as? NSNumber,
        CFGetTypeID(seed) != CFBooleanGetTypeID(),
        seed.doubleValue.isFinite,
        (0...Double(UInt32.max)).contains(seed.doubleValue),
        seed.doubleValue.rounded() == seed.doubleValue else {
        throw LTXError.invalid("Each Swift LTX scene shot needs a unique ID, prompt, finite duration and 32-bit seed.")
      }
      ids.append(id); prompts.append(prompt); durations.append(duration.doubleValue)
      seeds.append(seed.intValue)
    }
    guard Set(ids).count == ids.count else {
      throw LTXError.invalid("Swift LTX scene shot IDs must be unique.")
    }
    let fps = fpsNumber.doubleValue
    let plan = try LTX25ScenePlan(durations: durations, fps: fps,
      overlapFrames: overlap.intValue)
    guard durationNumber.doubleValue.isFinite,
      abs(durationNumber.doubleValue - Double(plan.totalFrames - 1) / fps) <= 1e-6 else {
      throw LTXError.invalid("Swift LTX scene duration does not match its exact frame plan.")
    }
    var requests: [MLXDistilledRequest] = []
    for index in entries.indices {
      var ordinary = root
      ordinary.removeValue(forKey: "scene")
      ordinary["prompt"] = prompts[index]
      var windowConfig = config
      windowConfig["duration_seconds"] = Double(plan.windowFrames[index] - 1) / fps
      windowConfig["seed"] = seeds[index]
      ordinary["config"] = windowConfig
      if index > 0 {
        var laterConditioning = conditioning
        var image: [String: Any]?
        if let imageInput = entries[index]["image_input"] {
          guard let validatedImage = imageInput as? [String: Any],
            validatedImage["kind"] as? String == "image",
            validatedImage["role"] as? String == "keyframe",
            validatedImage["frame_index"] as? Int == 0 else {
            throw LTXError.invalid("A later scene image must be a first-frame keyframe for its shot.")
          }
          image = validatedImage
        }
        if sceneTask == "a2v" {
          guard var driver = sceneInputs.first(where: { $0["role"] as? String == "audio_driver" }),
            let originalStart = driver["source_start_seconds"] as? Double else {
            throw LTXError.invalid("Swift LTX scene audio driver has no source in-point.")
          }
          driver["source_start_seconds"] = originalStart + Double(plan.windowStarts[index]) / fps
          driver["source_duration_seconds"] = Double(plan.totalFrames - 1 - plan.windowStarts[index]) / fps
          laterConditioning["task"] = "a2v"
          laterConditioning["inputs"] = [driver] + (image.map { [$0] } ?? [])
        } else {
          laterConditioning["task"] = image == nil ? "t2v" : "fflf"
          laterConditioning["inputs"] = image.map { [$0] } ?? []
        }
        ordinary["conditioning"] = laterConditioning
      } else if entries[index]["image_input"] != nil {
        throw LTXError.invalid("The first scene image belongs in the opening conditioning contract.")
      }
      let windowData = try JSONSerialization.data(withJSONObject: ordinary)
      let request = try MLXStudioRecipe.compile(data: windowData,
        outputDirectory: outputDirectory + "/window-\(index)")
      guard request.frames == plan.windowFrames[index],
        request.task == (sceneTask == "a2v" ? "a2v" :
          ((index == 0 ? sceneTask == "fflf" : entries[index]["image_input"] != nil) ? "i2v" : "t2v")),
        request.referenceImages.count <= 1,
        request.noisePolicy == .releasedMLX else {
        throw LTXError.invalid("Swift LTX scene window changed its validated sampling contract.")
      }
      requests.append(request)
    }
    // A full-scene VAE can pull a new image into earlier frames even when its
    // latent join is hard. Decode only at explicit image cuts (or every join
    // under strict policy), retaining ordinary continuity within each group.
    let imageBoundaries = Set((1..<entries.count).filter { entries[$0]["image_input"] != nil })
    let strictBoundaries = policy == "strict"
      ? Set(1..<entries.count) : imageBoundaries
    return Compiled(plan: plan, requests: requests, clipIDs: ids,
      decodeMode: decodeMode, strictBoundaries: strictBoundaries)
  }
}
