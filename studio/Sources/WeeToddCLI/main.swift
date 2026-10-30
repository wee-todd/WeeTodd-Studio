import Darwin
import Foundation
import StudioCore

let arguments = Array(CommandLine.arguments.dropFirst())
if arguments.first == "models",arguments.dropFirst().contains(where:{["help","--help","-h"].contains($0)}) {
  print("""
  WeeToddCLI models list [--catalog FILE]
  WeeToddCLI models download --package ID --model-library DIR [--catalog FILE] [--reuse-root DIR]
  WeeToddCLI models convert --profile tiny|wan|prompt --source FILE --model-library DIR
  WeeToddCLI models sharp-table --model-library DIR [--source BASE_SAFETENSORS]
  WeeToddCLI models prepare-h3 --model-library DIR [--catalog FILE] [--worker FILE] [--profile-directory DIR] [--include-fast true|false]
  WeeToddCLI models prepare-h3 --profile ref2va-base --model-library DIR [--source FILE] [--worker FILE]
  WeeToddCLI models prepare-h3 --profile publisher-encoder --model-library DIR [--source SHARD2 --source SHARD3] [--worker FILE]
  Existing verified prepared models are reused. New weights stay in the selected model library.
  """)
  exit(0)
}
if let mode=arguments.first,["regenerate","process"].contains(mode) {
  do {
    let task=Task {try await NativeRestorationCLI.run(mode:mode,arguments:Array(arguments.dropFirst()))}
    signal(SIGINT,SIG_IGN);signal(SIGTERM,SIG_IGN)
    let signals=[SIGINT,SIGTERM].map { number -> DispatchSourceSignal in
      let source=DispatchSource.makeSignalSource(signal:number,queue:.global())
      source.setEventHandler { @Sendable in task.cancel() };source.resume();return source
    }
    do {try await task.value;withExtendedLifetime(signals){exit(0)}}
    catch is CancellationError {withExtendedLifetime(signals){exit(130)}}
  } catch {fputs(error.localizedDescription+"\n",stderr);exit(1)}
}
if arguments.first == "models" {
  do {
    let options = try NativeUpscalerSetupArguments(Array(arguments.dropFirst()))
    let task = Task {
      let catalog = options.values["--catalog"].map { URL(fileURLWithPath: $0) } ?? NativeModelDownloads.bundledCatalog
      func printJSON<T: Encodable>(_ value: T) throws {
        FileHandle.standardOutput.write(try JSONEncoder().encode(value) + Data([10]))
      }
      func downloadToken() async throws -> String? {
        try await BackgroundCredential.read { try ModelDownloadToken.resolveExisting(environment:ProcessInfo.processInfo.environment) }
      }
      switch options.action {
      case .list:
        guard let catalog else { throw StudioError.invalid("Pass --catalog for a command-line build outside the packaged app.") }
        try printJSON(NativeModelDownloads.catalog(at: catalog).filter { ["h3-regenerate", "h3-fast-vae", "h3-encoder-source", "h3-ref2va-source", "flashvsr-native"].contains($0.kind) })
      case .download:
        guard let catalog else { throw StudioError.invalid("Pass --catalog for a command-line build outside the packaged app.") }
        let output = try await NativeUpscalerSetup.download(packageID: options.values["--package"]!, catalog: catalog,
          modelLibrary: URL(fileURLWithPath: options.values["--model-library"]!), reuseRoots: options.sources, token: try await downloadToken(),
          progress: { message, _ in FileHandle.standardError.write(Data((message + "\n").utf8)) })
        try printJSON(["status": "success", "modelPath": output.path, "runtime": "swift"])
      case .convert:
        try printJSON(NativeUpscalerSetup.convert(profile: options.values["--profile"]!,
          source: URL(fileURLWithPath: options.values["--source"]!), modelLibrary: URL(fileURLWithPath: options.values["--model-library"]!)))
      case .sharpTable:
        let output = try await NativeUpscalerSetup.prepareSharpTable(modelLibrary: URL(fileURLWithPath: options.values["--model-library"]!),
          existingSources: options.sources.map { URL(fileURLWithPath: $0) })
        try printJSON(["status": "success", "sharpTablePath": output.path, "runtime": "swift"])
      case .prepareH3:
        let library=URL(fileURLWithPath:options.values["--model-library"]!)
        let worker=options.values["--worker"].map{URL(fileURLWithPath:$0)}
        let progress:NativeModelDownloads.Progress={message,_ in FileHandle.standardError.write(Data((message+"\n").utf8))}
        switch options.values["--profile"] ?? "restoration" {
        case "ref2va-base":
          try printJSON(try await NativeUpscalerSetup.prepareH3Base(source:options.sources.first.map{URL(fileURLWithPath:$0)},modelLibrary:library,worker:worker,progress:progress))
        case "publisher-encoder":
          try printJSON(try await NativeUpscalerSetup.prepareH3Encoder(shards:options.sources.map{URL(fileURLWithPath:$0)},modelLibrary:library,worker:worker,progress:progress))
        default:
          guard let catalog else {throw StudioError.invalid("Pass --catalog for a command-line build outside the packaged app.")}
          let executable=worker ?? Bundle.main.executableURL?.deletingLastPathComponent().appendingPathComponent("WeeToddH3MLXWorker")
          guard let executable else {throw StudioError.invalid("Pass --worker for native H3 preparation.")}
          let result=try await NativeUpscalerSetup.prepareH3Restoration(catalog:catalog,modelLibrary:library,
            profileDirectory:options.values["--profile-directory"].map{URL(fileURLWithPath:$0)} ?? library.appendingPathComponent("WeeTodd/H3/profiles"),
            workerPath:executable.path,includeFastDecoder:options.values["--include-fast"] == "true",token:try await downloadToken(),progress:progress)
          try printJSON(["status":"success","profilePath":result.profileURL.path,"publisherEncoderPath":result.publisherEncoderURL.path,"fastDecoderPath":result.fastDecoderURL?.path ?? "","runtime":"swift"])
        }
      }
    }
    signal(SIGINT, SIG_IGN); signal(SIGTERM, SIG_IGN)
    let signals = [SIGINT, SIGTERM].map { number -> DispatchSourceSignal in
      let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
      source.setEventHandler { @Sendable in task.cancel() }; source.resume(); return source
    }
    do { try await task.value; withExtendedLifetime(signals) { exit(0) } }
    catch is CancellationError { withExtendedLifetime(signals) { exit(130) } }
  } catch { fputs(error.localizedDescription + "\n", stderr); exit(1) }
}
if arguments.first == "--flashvsr-engine-host" {
  do {
    guard arguments.count == 2 else { throw StudioError.invalid("An engine command file is required.") }
    exit(try FlashVSREngineSupervisor.run(commandURL: URL(fileURLWithPath: arguments[1])))
  } catch is CancellationError { exit(130) }
  catch { fputs(error.localizedDescription + "\n", stderr); exit(1) }
}
if arguments.first == "upscale" {
  do {
    let options = try FlashVSRCLIArguments(Array(arguments.dropFirst()))
    let url = URL(fileURLWithPath: options.requestPath)
    let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
    guard let size = attributes[.size] as? NSNumber, size.intValue > 0, size.intValue <= 1024 * 1024 else {
      throw StudioError.invalid("An upscale request must be a local JSON file of at most 1 MiB.")
    }
    var request = try JSONDecoder().decode(VideoUpscaleRequest.self, from: Data(contentsOf: url))
    if request.runtime.supervisorPath.isEmpty { request.runtime.supervisorPath = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL.path }
    let task = Task {
      let plan = try await FlashVSRService.preflight(request: request)
      if options.preflightOnly {
        let body = try JSONSerialization.jsonObject(with: JSONEncoder().encode(plan))
        FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: ["status": "success", "preflight": true, "plan": body]) + Data([10]))
      } else {
        let directory = URL(fileURLWithPath: options.outputDirectory)
        let receipt: FlashVSRReceipt
        if options.resume { receipt = try FlashVSRService.resume(plan: plan, directory: directory) }
        else { receipt = try await FlashVSRService.run(plan: plan, directory: directory, emit: { event in
          if let data = try? JSONEncoder().encode(event) { FileHandle.standardOutput.write(data + Data([10])) }
        }) }
        let body = try JSONSerialization.jsonObject(with: JSONEncoder().encode(receipt))
        FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: ["status": "success", "result": body]) + Data([10]))
      }
    }
    signal(SIGINT, SIG_IGN); signal(SIGTERM, SIG_IGN)
    let signals = [SIGINT, SIGTERM].map { number -> DispatchSourceSignal in
      let source = DispatchSource.makeSignalSource(signal: number, queue: .global())
      source.setEventHandler { task.cancel() }; source.resume(); return source
    }
    do { try await task.value; withExtendedLifetime(signals) { exit(0) } }
    catch is CancellationError { withExtendedLifetime(signals) { exit(130) } }
  } catch { fputs(error.localizedDescription + "\n", stderr); exit(1) }
}
if arguments.first == "--audio-presentation" {
  do { try AudioPresentationExport.run(arguments: Array(arguments.dropFirst())); exit(0) }
  catch { fputs(error.localizedDescription + "\n", stderr); exit(1) }
}
if arguments.isEmpty || arguments.contains("--help") {
  print(
    """
    WeeTodd Studio headless movie and clip renderer

    WeeToddCLI upscale --request Upscale.json --output-directory Upscaled [--preflight-only] [--resume]
    WeeToddCLI models list [--catalog FILE]
    WeeToddCLI models download --package ID --model-library DIR [--catalog FILE] [--reuse-root DIR]
    WeeToddCLI models convert --profile tiny|wan|prompt --source FILE --model-library DIR
    WeeToddCLI models sharp-table --model-library DIR [--source BASE_SAFETENSORS]
    WeeToddCLI models prepare-h3 --model-library DIR [--catalog FILE] [--worker FILE] [--profile-directory DIR] [--include-fast true|false]
    WeeToddCLI models prepare-h3 --profile ref2va-base|publisher-encoder --model-library DIR [--source FILE] [--source SECOND_SHARD] [--worker FILE]
    WeeToddCLI regenerate --request H3.json --output-directory Restored [--preflight-only]
    WeeToddCLI process --request Processing.json --output-directory Cache [--preflight-only] [--resume]
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
  if NativeHeadlessJob.supportedFormats.contains(raw?["format"] as? String ?? "") {
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
