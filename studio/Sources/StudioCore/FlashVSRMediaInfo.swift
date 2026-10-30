import Foundation

public struct FlashVSRFrameRate: Codable, Equatable, Sendable {
  public let numerator: Int, denominator: Int
  public var value: Double { Double(numerator) / Double(denominator) }
  public var argument: String { "\(numerator)/\(denominator)" }
  public init(numerator: Int, denominator: Int) throws {
    guard numerator > 0, denominator > 0, numerator <= 1_000_000, denominator <= 1_000_000,
      (1...120).contains(Double(numerator) / Double(denominator)) else {
      throw StudioError.invalid("FlashVSR+ requires a finite constant frame rate from 1 to 120 FPS.")
    }
    self.numerator = numerator; self.denominator = denominator
  }
}
public struct FlashVSRMediaInfo: Codable, Equatable, Sendable {
  public let width: Int, height: Int, frames: Int
  public let fps: FlashVSRFrameRate
  public let firstTimestamp: Double
  public let hasAudio: Bool
  public let audioRate: Int?, audioChannels: Int?
  public static func parse(summary: Data, timestamps: Data, expectedFrameRate: FlashVSRFrameRate? = nil) throws -> Self {
    guard let body = try JSONSerialization.jsonObject(with: summary) as? [String: Any],
      let streams = body["streams"] as? [[String: Any]] else { throw StudioError.invalid("Cannot inspect FlashVSR+ source streams.") }
    let videos = streams.filter { $0["codec_type"] as? String == "video" }
    guard videos.count == 1, let video = videos.first, let width = video["width"] as? Int,
      let height = video["height"] as? Int, (1...8192).contains(width), (1...8192).contains(height),
      let rate = video["avg_frame_rate"] as? String else { throw StudioError.invalid("Select a movie with one supported video track.") }
    let parts = rate.split(separator: "/").compactMap { Int($0) }
    guard parts.count == 2 else { throw StudioError.invalid("The source movie has no rational frame rate.") }
    let declared = try? FlashVSRFrameRate(numerator: parts[0], denominator: parts[1])
    let transfer = video["color_transfer"] as? String ?? "unknown"
    let pixel = video["pix_fmt"] as? String ?? ""
    guard !["smpte2084", "arib-std-b67", "linear"].contains(transfer),
      !["yuva", "rgba", "bgra", "argb", "abgr", "gbrap", "ya8", "ya16"].contains(where: { pixel.hasPrefix($0) }),
      !pixel.contains("p10"), !pixel.contains("p12"), !pixel.contains("p16"),
      ["1:1", "N/A"].contains(video["sample_aspect_ratio"] as? String ?? "1:1") else {
      throw StudioError.invalid("FlashVSR+ currently accepts SDR movies with square pixels and no alpha. Normalize HDR, alpha or pixel aspect explicitly first.")
    }
    let records = String(decoding: timestamps, as: UTF8.self).split(whereSeparator: \.isNewline)
    let times = try records.map { line -> Double in
      guard let value = Double(line.split(separator: ",", omittingEmptySubsequences: false).first ?? ""), value.isFinite else {
        throw StudioError.invalid("A video frame has no valid presentation timestamp. Its timing cannot be preserved.")
      }
      return value
    }
    // Raw NUT can omit average cadence. Only an internal restoration may
    // use its frozen clock, and every observed timestamp must still agree.
    guard let fps = declared ?? expectedFrameRate else {
      throw StudioError.invalid("The movie has no valid declared constant frame rate.")
    }
    guard let first = times.first, first.isFinite, times.count <= 100_000,
      times.enumerated().allSatisfy({ $0.element.isFinite && abs(($0.element - first) * fps.value - Double($0.offset)) < 0.05 }) else {
      throw StudioError.invalid("The source has variable or invalid frame timing. Normalize it to constant FPS before upscaling.")
    }
    let audio = streams.first { $0["codec_type"] as? String == "audio" }
    // FFmpeg's normal decode autorotates display matrices. Admission must use
    // that same visible canvas, while retaining the untouched source clock.
    let sideData=video["side_data_list"] as? [[String:Any]] ?? []
    let matrices=sideData.filter{$0["side_data_type"] as? String == "Display Matrix" || $0["rotation"] != nil}
    guard matrices.count<=1 else{throw StudioError.invalid("The source has ambiguous display rotation.")}
    let angle:Double
    if let matrix=matrices.first {
      guard let value=matrix["rotation"] as? NSNumber else{throw StudioError.invalid("The source display rotation is invalid.")}
      angle=value.doubleValue
    } else if let raw=(video["tags"] as? [String:Any])?["rotate"] {
      guard let text=raw as? String,let value=Double(text) else{throw StudioError.invalid("The source display rotation is invalid.")}
      angle=value
    } else {angle=0}
    guard angle.isFinite,abs(angle)<=3600,abs(angle/90-(angle/90).rounded())<0.000001 else{
      throw StudioError.invalid("Only right-angle source display rotation can be preserved. Normalize other display transforms explicitly before upscaling.")
    }
    let swapsAxes=abs(Int((angle/90).rounded()))%2==1
    return Self(width: swapsAxes ? height:width, height: swapsAxes ? width:height, frames: times.count, fps: fps, firstTimestamp: first,
      hasAudio: audio != nil, audioRate: (audio?["sample_rate"] as? String).flatMap(Int.init), audioChannels: audio?["channels"] as? Int)
  }
  static func inspect(_ url: URL, runtime: FlashVSRRuntime, directory: URL,
    cancellation: NativeHeadlessCancellation, expectedFrameRate: FlashVSRFrameRate? = nil) throws -> Self {
    let id = UUID().uuidString
    let summary = try ExternalEngineProcess.run(executable: URL(fileURLWithPath: runtime.ffprobePath),
      arguments: ["-v", "error", "-show_streams", "-of", "json", url.path], directory: directory,
      log: directory.appendingPathComponent(id + "-streams.log"), cancellation: cancellation)
    guard summary.exitCode == 0 else { throw StudioError.invalid("FFprobe could not read the movie streams.") }
    let timing = try ExternalEngineProcess.run(executable: URL(fileURLWithPath: runtime.ffprobePath),
      arguments: ["-v", "error", "-select_streams", "v:0", "-show_frames", "-show_entries", "frame=best_effort_timestamp_time:frame_side_data=", "-of", "csv=p=0", url.path], directory: directory,
      log: directory.appendingPathComponent(id + "-timing.log"), cancellation: cancellation, maximumOutput: 8 * 1024 * 1024)
    guard timing.exitCode == 0 else { throw StudioError.invalid("FFprobe could not read the complete movie clock.") }
    return try parse(summary: summary.output, timestamps: timing.output, expectedFrameRate: expectedFrameRate)
  }
}
