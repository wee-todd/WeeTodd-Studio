import Foundation
import CoreFoundation

/// Binds complete-take timing to frozen automatic intent and actual video geometry.
public enum NativeLTXAutomaticDurationAcceptance {
  public static func duration(prepared: [String:Any]?, result: [String:Any],
    measuredVideoDuration: Double, measuredContainerDuration: Double, measuredFPS: Double) throws -> Double? {
    guard prepared?["durationMode"] as? String == "automatic" else { return nil }
    func numeric(_ raw: Any?) -> Double? {
      guard let value = raw as? NSNumber, CFGetTypeID(value) != CFBooleanGetTypeID(), value.doubleValue.isFinite else { return nil }
      return value.doubleValue
    }
    func fail() -> StudioError { .invalid("Automatic duration result does not match the frozen head policy or inspected video geometry. The take remains saved for review.") }
    guard let frozen = prepared?["automaticDuration"] as? [String:Any],
      let metadata = result["metadata"] as? [String:Any], let actual = metadata["automatic_duration"] as? [String:Any],
      result["nativeRuntime"] as? String == "swift-mlx", result["use_complete_duration"] as? Bool == true,
      let head = frozen["head_checkpoint_path"] as? String, head.hasPrefix("/"),
      actual["head_checkpoint_path"] as? String == head,
      let sha = frozen["head_header_sha256"] as? String, sha.utf8.count == 64,
      sha.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }),
      actual["head_header_sha256"] as? String == sha,
      let minimum = numeric(frozen["minimum_seconds"]), let maximum = numeric(frozen["maximum_seconds"]),
      numeric(actual["minimum_seconds"]) == minimum, numeric(actual["maximum_seconds"]) == maximum,
      let fps = numeric(actual["fps"]), numeric(prepared?["nativeFPS"]) == fps,
      numeric(metadata["fps"]) == fps,
      let rawFrames = numeric(actual["resolved_frames"]), rawFrames.rounded() == rawFrames,
      (1...3601).contains(rawFrames), numeric(metadata["frames"]) == rawFrames,
      let prediction = numeric(actual["predicted_duration_seconds"]), prediction > 0 else { throw fail() }
    let frames = Int(rawFrames)
    let maximumFrames = try NativeLTXAutomaticDuration.maximumFrames(minimumSeconds:minimum,maximumSeconds:maximum,fps:fps)
    let minimumFrame = Int((minimum*fps).rounded(.toNearestOrEven))
    guard frames%8 == 1, frames >= minimumFrame, frames <= maximumFrames else { throw fail() }
    let duration = rawFrames/fps, tolerance = 0.5/fps
    guard measuredVideoDuration.isFinite, measuredContainerDuration.isFinite, measuredFPS.isFinite,
      abs(measuredFPS-fps) <= 0.001, abs(measuredVideoDuration-duration) <= tolerance,
      measuredContainerDuration+tolerance >= duration,
      let usable = numeric(result["usable_duration"]), abs(usable-duration) <= 0.000001,
      (numeric(result["usable_source_in"]) ?? 0) == 0,
      let reportedDuration = numeric(metadata["video_seconds"]), abs(reportedDuration-duration) <= 0.000001 else { throw fail() }
    return duration
  }
}
