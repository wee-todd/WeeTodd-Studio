import Foundation

enum FlashVSRAudioPreparation {
  static func prepare(source: URL, output: URL, ffmpeg: URL, directory: URL, startTimestamp: Double,
    frames: Int, fps: FlashVSRFrameRate, sampleRate: Int, cancellation: NativeHeadlessCancellation) throws {
    let samples = Int64((Double(frames) / fps.value * Double(sampleRate)).rounded())
    // Anchor to the video interval rather than the first surviving audio packet.
    // first_pts inserts leading silence; apad also covers intervals with no packets.
    let filter = "atrim=start=\(startTimestamp):duration=\(Double(frames) / fps.value),asetpts=PTS-\(startTimestamp)/TB,aresample=first_pts=0,apad=whole_len=\(samples),atrim=end_sample=\(samples)"
    let result = try ExternalEngineProcess.run(executable: ffmpeg, arguments: ["-v", "error", "-nostdin", "-n",
      "-copyts", "-i", source.path, "-map", "0:a:0", "-vn", "-af", filter, "-c:a", "pcm_f32le", output.path],
      directory: directory, log: output.deletingPathExtension().appendingPathExtension("log"), cancellation: cancellation)
    guard result.exitCode == 0 else { throw StudioError.invalid("FlashVSR+ source audio preparation failed.") }
  }
}
