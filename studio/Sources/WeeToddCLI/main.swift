import Darwin
import Foundation
import StudioCore

let arguments = Array(CommandLine.arguments.dropFirst())
if arguments.isEmpty || arguments.contains("--help") {
  print(
    """
    WeeTodd Studio headless movie and clip renderer

    WeeToddCLI --job Movie.weetodd-job.json --output-directory Render --preflight-only
    WeeToddCLI --job Movie.weetodd-job.json --output-directory Render --resume

    Uses the runtime recorded in the job. Native H3/LTX jobs execute without Python.
    Override a moved worker with --h3-swift-worker PATH or --ltx25-swift-worker PATH.
    The graphical editor can be closed.
    """)
  exit(0)
}
do {
  guard let index = arguments.firstIndex(of: "--job"), arguments.count > index + 1 else {
    throw NSError(
      domain: "WeeToddCLI", code: 1,
      userInfo: [NSLocalizedDescriptionKey: "Pass --job followed by an exported movie or clip job."]
    )
  }
  let url = URL(fileURLWithPath: arguments[index + 1])
  let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
  if raw?["format"] as? String == NativeHeadlessJob.format {
    let options = try NativeHeadlessCLIArguments(arguments)
    let job = try NativeHeadlessJob.read(from: URL(fileURLWithPath: options.jobPath),
      workerOverrides: options.workerOverrides)
    let cancellation = NativeHeadlessCancellation()
    signal(SIGINT, SIG_IGN); signal(SIGTERM, SIG_IGN)
    let interrupts = [SIGINT, SIGTERM].map { number in
      let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
      source.setEventHandler { cancellation.cancel() }; source.resume(); return source
    }
    do {
      let result = try await NativeHeadlessExecutor.run(job: job,
        output: URL(fileURLWithPath: options.outputDirectory), resume: options.resume,
        preflightOnly: options.preflightOnly, cancellation: cancellation)
      FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: result) + Data([10]))
      withExtendedLifetime(interrupts) { exit(0) }
    } catch is CancellationError {
      print("{\"status\":\"cancelled\",\"native_runtime\":\"swift-mlx\"}")
      withExtendedLifetime(interrupts) { exit(130) }
    }
  }
  guard let job = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any],
    ["weetodd-studio-job-v1", "weetodd-studio-job-v2", "weetodd-studio-job-v3", "weetodd-studio-job-v4"].contains(
      job["format"] as? String ?? ""),
    let runtime = job["runtime"] as? [String: Any], let python = runtime["pythonPath"] as? String,
    let root = runtime["root"] as? String,
    FileManager.default.isExecutableFile(atPath: python),
    FileManager.default.fileExists(atPath: root + "/scripts/render_headless.py")
  else {
    throw NSError(
      domain: "WeeToddCLI", code: 1,
      userInfo: [
        NSLocalizedDescriptionKey:
          "The job's native runtime is unavailable. Open Studio, connect its runtime, and export the job again."
      ])
  }
  let child = Process()
  child.executableURL = URL(fileURLWithPath: python)
  child.arguments = [root + "/scripts/render_headless.py"] + arguments
  child.standardInput = FileHandle.standardInput
  child.standardOutput = FileHandle.standardOutput
  child.standardError = FileHandle.standardError
  var env = ProcessInfo.processInfo.environment
  env["PYTHONUNBUFFERED"] = "1"
  child.environment = env
  signal(SIGINT, SIG_IGN)
  signal(SIGTERM, SIG_IGN)
  let interrupts = [SIGINT, SIGTERM].map { number in
    let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
    source.setEventHandler { if child.isRunning { child.interrupt() } }
    source.resume()
    return source
  }
  try child.run()
  child.waitUntilExit()
  withExtendedLifetime(interrupts) { exit(child.terminationStatus) }
} catch {
  fputs(error.localizedDescription + "\n", stderr)
  exit(1)
}
