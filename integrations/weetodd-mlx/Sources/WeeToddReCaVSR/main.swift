import Darwin
import Foundation
import InferenceContracts
import MLX
import ReCaVSRMLX

@main struct WeeToddReCaVSR {
  static let usage = """
    WeeToddReCaVSR INPUT OUTPUT.mov --models DIRECTORY
      [--scale 4|2|1.5] [--seed 42] [--precision bf16|int8ConvRot]
      [--color-fix none|adain|wavelet] [--residency staged|resident]
      [--source-start SECONDS --source-duration SECONDS] [--frames LIMIT]
      [--budget-gib 32] [--reset-frames 100,200] [--preflight]
      [--capture-rgb NEW_ABSOLUTE_DIRECTORY] [--attention-mode auto|sync]
    WeeToddReCaVSR --recipe ABSOLUTE_JSON [--output OUTPUT.mov] [--preflight]
    Source presentation timestamps and audio are preserved; no frame interpolation.
    """
  static func emit(_ object: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: object), data.count < 65536 else { return }
    let bytes = data + Data([10])
    bytes.withUnsafeBytes { raw in
      var offset = 0
      while offset < raw.count {
        let n = Darwin.write(STDOUT_FILENO, raw.baseAddress!.advanced(by: offset), raw.count - offset)
        if n > 0 { offset += n }
        else if errno == EINTR { continue }
        else { return }
      }
    }
  }
  static func main() async {
    signal(SIGPIPE, SIG_IGN)
    do {
      var args = Array(CommandLine.arguments.dropFirst())
      if args.isEmpty || args == ["--help"] { print(usage); return }
      let studio = args.contains("--studio"), preflight = args.contains("--preflight")
      args.removeAll { $0 == "--studio" || $0 == "--preflight" }
      let recipeMode = args.first == "--recipe"
      let offset = recipeMode ? 0 : 2
      guard args.count >= offset, (args.count - offset).isMultiple(of: 2) else { throw ReCaError.invalid(usage) }
      var flags: [String: String] = [:]
      for i in stride(from: offset, to: args.count, by: 2) {
        guard flags[args[i]] == nil else { throw ReCaError.invalid("Repeated option \(args[i]).") }
        flags[args[i]] = args[i + 1]
      }
      let allowed: Set<String> = recipeMode ? ["--recipe", "--output"] : ["--models", "--scale", "--seed", "--precision", "--color-fix", "--residency", "--source-start", "--source-duration", "--frames", "--budget-gib", "--reset-frames"]
      let diagnosticAllowed: Set<String> = ["--capture-rgb", "--attention-mode"]
      guard Set(flags.keys).isSubset(of: allowed.union(diagnosticAllowed)) else { throw ReCaError.invalid("Unknown ReCaVSR option.\n" + usage) }
      guard let attentionMode = ReCaAttentionScheduling(rawValue:flags["--attention-mode"] ?? "auto") else {
        throw ReCaError.invalid("Attention mode must be auto or sync.")
      }
      let captureDirectory: URL?
      if let raw = flags["--capture-rgb"] {
        guard raw.hasPrefix("/"), !FileManager.default.fileExists(atPath:raw),
          FileManager.default.fileExists(atPath:URL(fileURLWithPath:raw).deletingLastPathComponent().path) else {
          throw ReCaError.invalid("RGB capture requires a new absolute directory with an existing parent.")
        }
        captureDirectory = URL(fileURLWithPath:raw)
      } else { captureDirectory = nil }
      let executable = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
      let library = try NativeMLXMetalLibrary.resolve(executable: executable)
      GPU.metallib = library
      var input: NativeReCaVSRRecipe
      if recipeMode {
        guard let file = flags["--recipe"], file.hasPrefix("/") else { throw ReCaError.invalid("Provide an absolute recipe path.") }
        let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: file)); defer { try? handle.close() }
        input = try NativeReCaVSRRecipe.prepare(handle.read(upToCount: 1024 * 1024 + 1) ?? Data())
        let runningWorker = try ReCaFileIdentity.capture(executable)
        let runningMetal = try ReCaFileIdentity.capture(library)
        guard input.worker.sha256 == runningWorker.sha256, input.worker.bytes == runningWorker.bytes,
          input.metal.sha256 == runningMetal.sha256, input.metal.bytes == runningMetal.bytes else {
          throw ReCaError.invalid("Running ReCaVSR worker or Metal library identity changed. Prepare the recipe again.")
        }
        if let output = flags["--output"] { input.outputPath = URL(fileURLWithPath: output).standardizedFileURL.path }
      } else {
        guard args.count >= 2, let directory = flags["--models"] else { throw ReCaError.invalid(usage) }
        func number(_ flag: String, fallback: Double) throws -> Double {
          guard let raw = flags[flag] else { return fallback }
          guard let value = Double(raw), value.isFinite else { throw ReCaError.invalid("Invalid \(flag).") }; return value
        }
        guard let precision = ReCaPrecision(rawValue: flags["--precision"] ?? "bf16"),
          let color = ReCaColorFix(rawValue: flags["--color-fix"] ?? "none"),
          let residency = ReCaResidency(rawValue: flags["--residency"] ?? "staged"),
          let seed = UInt64(flags["--seed"] ?? "42") else { throw ReCaError.invalid("Invalid precision, color, residency or seed.") }
        let settings = try ReCaVSRSettings(scale: number("--scale", fallback: 4), seed: seed,
          precision: precision, colorFix: color, residency: residency)
        let interval: ReCaSourceInterval?
        if flags["--source-start"] != nil || flags["--source-duration"] != nil {
          guard flags["--source-duration"] != nil else { throw ReCaError.invalid("A source interval requires --source-duration.") }
          interval = try ReCaSourceInterval(startSeconds: number("--source-start", fallback: 0), durationSeconds: number("--source-duration", fallback: 0))
        } else { interval = nil }
        let frameLimit = flags["--frames"].flatMap(Int.init)
        guard flags["--frames"] == nil || frameLimit != nil else { throw ReCaError.invalid("Invalid frame limit.") }
        let budget = try number("--budget-gib", fallback: 32)
        guard budget >= 1, budget <= 1024 else { throw ReCaError.invalid("Memory budget must be 1…1024 GiB.") }
        let resets = try (flags["--reset-frames"]?.split(separator: ",") ?? []).map { value -> Int in
          guard let frame = Int(value) else { throw ReCaError.invalid("Invalid reset frame.") }; return frame
        }
        input = try NativeReCaVSRRecipe(source: ReCaFileIdentity.capture(URL(fileURLWithPath: args[0])),
          interval: interval, worker: ReCaFileIdentity.capture(executable), metal: ReCaFileIdentity.capture(library),
          assets: ReCaReleasedAssets.resolve(directory: URL(fileURLWithPath: directory), precision: precision), settings: settings,
          outputPath: URL(fileURLWithPath: args[1]).standardizedFileURL.path,
          maximumActiveBytes: UInt64(budget * Double(1 << 30)), frameLimit: frameLimit, resetFrames: resets)
      }
      let recipe = input
      signal(SIGINT, SIG_IGN); signal(SIGTERM, SIG_IGN)
      let flagsOut = fcntl(STDOUT_FILENO, F_GETFL)
      if flagsOut >= 0 { _ = fcntl(STDOUT_FILENO, F_SETFL, flagsOut | O_NONBLOCK) }
      let task = Task.detached {
        if preflight {
          let plan = try await ReCaMovieRunner.preflight(recipe)
          emit(["status": "ok", "result": ["preflight": true, "inference": false, "estimatedBytes": plan.estimatedBytes]])
        } else {
          let result = try await ReCaMovieRunner.run(recipe: recipe, attentionScheduling:attentionMode, captureRGBDirectory:captureDirectory) { progress in
            if studio { emit(["event": "progress", "message": progress.phase + " · \(progress.completedFrames) frames", "frames": progress.completedFrames]) }
            else { FileHandle.standardError.write(Data((progress.phase + " · \(progress.completedFrames) frames\n").utf8)) }
          }
          emit(["status": "ok", "result": ["video": result.video.path, "seconds": result.seconds,
            "receipt": result.video.path + ".json", "usable_source_in": result.sourceIn, "usable_duration": result.duration,
            "metadata": ["task": "video_upscale", "frames": result.frames, "model": "recavsr", "fidelity": result.fidelity]]])
        }
      }
      let signals = [SIGINT, SIGTERM].map { number -> DispatchSourceSignal in
        let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
        source.setEventHandler { @Sendable in task.cancel() }; source.resume(); return source
      }
      let owner = recipe.ownerPID ?? getppid(), parent = DispatchSource.makeTimerSource(queue: .global())
      parent.schedule(deadline: .now(), repeating: .milliseconds(200))
      parent.setEventHandler { @Sendable in
        if owner > 1, Darwin.kill(owner, 0) != 0, errno == ESRCH { task.cancel() }
        var descriptor = pollfd(fd: STDOUT_FILENO, events: Int16(POLLOUT), revents: 0)
        if poll(&descriptor, 1, 0) > 0, descriptor.revents & Int16(POLLERR | POLLHUP | POLLNVAL) != 0 { task.cancel() }
      }
      parent.resume(); defer { parent.cancel(); signals.forEach { $0.cancel() } }
      try await task.value
    } catch {
      emit(["status": error is CancellationError ? "cancelled" : "error", "error": error.localizedDescription])
      exit(EXIT_FAILURE)
    }
  }
}
