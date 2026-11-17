import Foundation
import MLX
import RelayVSR
import InferenceContracts
import Darwin

@main
struct WeeToddRelayVSR {
  private static let usage = """
    Usage: WeeToddRelayVSR INPUT OUTPUT.mov --models DIRECTORY
           [--scale 2|4] [--steps 1|4] [--seed INTEGER] [--interval FRAMES] [--detail 0…2] [--frames LIMIT]
           [--source-start SECONDS] [--source-duration SECONDS]
           [--preflight]

    Native RelayVSR 2× or 4× on Apple Silicon. Source timestamps and audio are preserved.
    """

  static func main() async {
    var arguments = Array(CommandLine.arguments.dropFirst())
    let studio = arguments.first == "--studio"
    if studio { arguments.removeFirst() }
    let preflight = arguments.contains("--preflight")
    arguments.removeAll { $0 == "--preflight" }
    do {
      if arguments.isEmpty || arguments == ["--help"] {
        print(usage)
        return
      }
      guard arguments.count >= 4, arguments.count.isMultiple(of: 2) else {
        throw RelayVSRError.invalid(usage)
      }
      var options = RelayMovieOptions()
      var models: URL?
      var seen = Set<String>()
      for index in stride(from: 2, to: arguments.count, by: 2) {
        let flag = arguments[index]
        let value = arguments[index + 1]
        guard seen.insert(flag).inserted else {
          throw RelayVSRError.invalid("Repeated option \(flag).")
        }
        switch flag {
        case "--scale":
          guard let number = Int(value), let scale = NativeRelayVSRScale(rawValue: number) else { throw RelayVSRError.invalid("RelayVSR supports native scale 2 or 4.") }
          options.scale = scale
        case "--models": models = URL(fileURLWithPath: value, isDirectory: true)
        case "--steps":
          guard let steps = Int(value) else { throw RelayVSRError.invalid("Invalid step count.") }
          options.steps = steps
        case "--seed":
          guard let seed = UInt64(value) else { throw RelayVSRError.invalid("Invalid unsigned seed.") }
          options.seed = seed
        case "--interval":
          guard let interval = Int(value) else { throw RelayVSRError.invalid("Invalid keyframe interval.") }
          options.keyframeInterval = interval
        case "--detail":
          guard let detail = Float(value) else { throw RelayVSRError.invalid("Invalid detail strength.") }
          options.detail = detail
        case "--frames":
          guard let limit = Int(value) else { throw RelayVSRError.invalid("Invalid frame limit.") }
          options.frameLimit = limit
        case "--source-start":
          guard let start = Double(value) else { throw RelayVSRError.invalid("Invalid source start.") }
          options.sourceStart = start
        case "--source-duration":
          guard let duration = Double(value) else { throw RelayVSRError.invalid("Invalid source duration.") }
          options.sourceDuration = duration
        default: throw RelayVSRError.invalid("Unknown option \(flag).")
        }
      }
      guard let models else { throw RelayVSRError.invalid("A --models directory is required.") }
      let executable = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
      let folder = executable.deletingLastPathComponent()
      if folder.lastPathComponent == "MacOS" {
        let library = folder.deletingLastPathComponent().appendingPathComponent("Resources/LTXNative/mlx.metallib")
        guard FileManager.default.isReadableFile(atPath: library.path) else {
          throw RelayVSRError.invalid("The packaged MLX Metal library is missing. Use a complete Studio build.")
        }
        GPU.metallib = library
      }
      let started = Date()
      if preflight {
        let admission = try await RelayMovie.preflight(input: URL(fileURLWithPath: arguments[0]), modelDirectory: models, options: options)
        let body = try JSONSerialization.jsonObject(with: JSONEncoder().encode(admission))
        emit(["status": "ok", "result": ["preflight": true, "inference": false, "admission": body]])
        return
      }
      let report: @Sendable (String) -> Void = { message in
        if studio { emit(["event": "progress", "message": message]) }
        else { FileHandle.standardError.write(Data((message + "\n").utf8)) }
      }
      let render = Task.detached { [arguments, models, options] in
        let movie = try await RelayMovie.upscale(input: URL(fileURLWithPath: arguments[0]),
          output: URL(fileURLWithPath: arguments[1]), modelDirectory: models, options: options,
          onWait: { report("Waiting for the native inference slot…") }) { count in
          if count == 1 || count.isMultiple(of: 15) { report("RelayVSR wrote \(count) frames") }
        }
        try Task.checkCancellation()
        return movie
      }
      signal(SIGINT, SIG_IGN); signal(SIGTERM, SIG_IGN)
      let signals = [SIGINT, SIGTERM].map { number -> DispatchSourceSignal in
        let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
        source.setEventHandler { @Sendable in render.cancel() }; source.resume(); return source
      }
      let owner = getppid(), parent = DispatchSource.makeTimerSource(queue: .global())
      parent.schedule(deadline: .now(), repeating: .milliseconds(200))
      parent.setEventHandler { @Sendable in
        if owner > 1, Darwin.kill(owner, 0) != 0, errno == ESRCH { render.cancel() }
      }
      parent.resume(); defer { parent.cancel(); signals.forEach { $0.cancel() } }
      let movie = try await render.value
      guard !render.isCancelled else { throw CancellationError() }
      let elapsed = Date().timeIntervalSince(started)
      if studio {
        emit(["status": "ok", "result": ["video": arguments[1], "seconds": elapsed,
          "usable_source_in": movie.sourceIn, "usable_duration": movie.duration,
          "metadata": ["task": "video_upscale", "frames": movie.frames, "model": "relayvsr-x\(options.scale.rawValue)", "scale": options.scale.rawValue]]])
      } else {
        print("Wrote \(arguments[1]) in \(elapsed) seconds")
      }
    } catch {
      if studio { emit(["status": "error", "error": error.localizedDescription]) }
      else { FileHandle.standardError.write(Data("RelayVSR: \(error.localizedDescription)\n".utf8)) }
      exit(EXIT_FAILURE)
    }
  }

  private static func emit(_ object: [String: Any]) {
    guard var data = try? JSONSerialization.data(withJSONObject: object) else { return }
    data.append(10)
    FileHandle.standardOutput.write(data)
  }
}
