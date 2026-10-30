import Darwin
import Foundation
import FlashVSRMLX
import InferenceContracts
import MLX

@main struct FlashVSRWorker {
  static func emit(_ object:[String:Any])throws {
    let data=try JSONSerialization.data(withJSONObject:object,options:.sortedKeys)
    guard data.count<=65535 else { throw invalid("Native worker event exceeds the 64 KiB protocol bound.") }
    let payload=data+Data([10]),deadline=ProcessInfo.processInfo.systemUptime+2
    try payload.withUnsafeBytes { bytes in
      var offset=0
      while offset<bytes.count {
        try Task.checkCancellation()
        let count=Darwin.write(STDOUT_FILENO,bytes.baseAddress!.advanced(by:offset),bytes.count-offset)
        if count>0 { offset += count;continue }
        let issue=errno
        if count<0 && issue==EINTR { continue }
        if count<0 && (issue==EAGAIN || issue==EWOULDBLOCK) {
          guard ProcessInfo.processInfo.systemUptime<deadline else { throw invalid("Native event consumer stopped reading progress.") }
          var descriptor=pollfd(fd:STDOUT_FILENO,events:Int16(POLLOUT),revents:0)
          _ = poll(&descriptor,1,100);continue
        }
        throw invalid("Native event output is no longer available.")
      }
    }
  }
  static func invalid(_ message: String) -> NSError {
    NSError(domain: "WeeToddFlashVSRWorker", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
  }
  static func boundedError(_ error:Error)->String {
    String(decoding:String(describing:error).utf8.prefix(8192),as:UTF8.self)
  }
  static func outputLost()->Bool {
    var descriptor=pollfd(fd:STDOUT_FILENO,events:Int16(POLLOUT),revents:0)
    return poll(&descriptor,1,0)>0 && descriptor.revents & Int16(POLLERR|POLLHUP|POLLNVAL) != 0
  }
  static func read(_ path: String) throws -> Data {
    guard path.hasPrefix("/"), !path.utf8.contains(0) else { throw invalid("An absolute native request path is required.") }
    let fd = Darwin.open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK | O_NOFOLLOW)
    guard fd >= 0 else { throw invalid("Cannot open the native request.") }
    let handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true); defer { try? handle.close() }
    var status = stat()
    guard fstat(fd, &status) == 0, status.st_mode & S_IFMT == S_IFREG,
      status.st_size > 0, status.st_size <= 1024 * 1024 else { throw invalid("Native requests must be regular files under 1 MiB.") }
    let data = try handle.read(upToCount: 1024 * 1024 + 1) ?? Data()
    guard data.count == status.st_size else { throw invalid("The native request changed while reading.") }
    return data
  }
  static func object<T: Encodable>(_ value: T) throws -> Any {
    try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
  }
  static func main() async {
    // Broken progress/output pipes throw EPIPE so the detached task unwinds its
    // model buffers, owned scratch and lease instead of signal-terminating.
    signal(SIGPIPE,SIG_IGN)
    do {
      let outputFlags=fcntl(STDOUT_FILENO,F_GETFL)
      guard outputFlags>=0,fcntl(STDOUT_FILENO,F_SETFL,outputFlags|O_NONBLOCK)==0 else { throw invalid("Cannot configure bounded native event output.") }
      let args = Array(CommandLine.arguments.dropFirst())
      guard args.count == 5, ["plan", "preflight", "render"].contains(args[0]),
        args[1] == "--request", args[3] == "--output" else {
        throw invalid("Usage: WeeToddFlashVSRWorker plan|preflight|render --request ENVELOPE --output NEW_DIRECTORY")
      }
      let executable=URL(fileURLWithPath:CommandLine.arguments[0])
      GPU.metallib=try NativeMLXMetalLibrary.resolve(executable:executable)
      let mode = args[0], output = URL(fileURLWithPath: args[4]).standardizedFileURL.resolvingSymlinksInPath()
      let envelope = try NativeVideoJobEnvelope.decode(read(args[2]), expectedEngine: .flashVSR, expectedOutputDirectory: output.path)
      let bytes = try read(envelope.recipePath); try envelope.validateRecipe(bytes)
      let recipe = try NativeFlashVSRRecipe.prepare(data: bytes)
      let raw = URL(fileURLWithPath: recipe.outputRGBPath).standardizedFileURL.resolvingSymlinksInPath()
      let scratch = URL(fileURLWithPath: recipe.scratchDirectory).standardizedFileURL.resolvingSymlinksInPath()
      var outputStatus=stat()
      let unused=lstat(output.path,&outputStatus) != 0 && errno==ENOENT
      guard raw.path.hasPrefix(output.path + "/"), scratch.path.hasPrefix(output.path + "/"),
        unused else { throw invalid("The native output must be a fresh owned directory containing its RGB output and scratch.") }
      signal(SIGINT,SIG_IGN); signal(SIGTERM,SIG_IGN)
      let task = Task.detached {
        try Task.checkCancellation()
        let owner = recipe.ownerPID ?? getppid()
        guard owner > 1, Darwin.kill(owner,0) == 0 || errno != ESRCH else { throw CancellationError() }
        if mode == "plan" {
          let plan = try FlashVSRMemoryPlan(recipe: recipe)
          try emit(["status":"success", "result":["jobID":envelope.jobID.uuidString, "nativeRuntime":"swift-mlx", "planOnly":true, "memory":try object(plan)]])
          return
        }
        if mode == "preflight" {
          let admission = try FlashVSRPipeline.preflight(recipe: recipe)
          try emit(["status":"success", "result":["jobID":envelope.jobID.uuidString, "nativeRuntime":"swift-mlx", "preflight":true, "admission":try object(admission)]])
          return
        }
        let lease = try NativeInferenceLease.acquire(at: recipe.leasePath.map { URL(fileURLWithPath:$0) } ?? NativeInferenceLease.defaultURL) {
          try? emit(["event":"progress", "stage":"waiting", "message":"Waiting for the native inference slot"])
        }
        defer { lease.release() }
        try Task.checkCancellation()
        try FileManager.default.createDirectory(at:output.deletingLastPathComponent(),withIntermediateDirectories:true)
        guard mkdir(output.path,S_IRWXU)==0 else { throw invalid("The native output directory already exists or cannot be created exclusively.") }
        var completed = false
        defer {
          if !completed {
            // Pipeline owns/removes its UUID scratch and exact partial file.
            // Never delete a pre-existing or already atomically completed RGB.
            // Empty-directory removal preserves any unrelated user files.
            _ = rmdir(scratch.path);_ = rmdir(output.path)
          }
        }
        let result = try FlashVSRPipeline.run(recipe:recipe) { progress in
          try emit(["event":"progress", "stage":progress.stage, "completed":progress.completed,
            "total":progress.total, "activeBytes":progress.activeBytes,
            "fraction":progress.total > 0 ? Double(progress.completed)/Double(progress.total):0])
        }
        try Task.checkCancellation()
        try emit(["status":"success", "result":["jobID":envelope.jobID.uuidString, "nativeRuntime":"swift-mlx",
          "video":raw.path, "pixelFormat":"rgb24", "metrics":try object(result)]])
        completed = true
      }
      let signals = [SIGINT,SIGTERM].map { number -> DispatchSourceSignal in
        let source = DispatchSource.makeSignalSource(signal:number,queue:.global())
        source.setEventHandler { @Sendable in task.cancel() }; source.resume(); return source
      }
      let parent = DispatchSource.makeTimerSource(queue:.global())
      let owner = recipe.ownerPID ?? getppid()
      parent.schedule(deadline:.now(),repeating:.milliseconds(200))
      parent.setEventHandler { @Sendable in
        if (Darwin.kill(owner,0) != 0 && errno==ESRCH) || outputLost() { task.cancel() }
      }
      parent.resume(); defer { parent.cancel(); signals.forEach { $0.cancel() } }
      try await task.value
      withExtendedLifetime(signals) {}
    } catch is CancellationError {
      try? emit(["status":"cancelled", "error":"Native restoration cancelled"])
      exit(130)
    } catch {
      try? emit(["status":"error", "error":boundedError(error)])
      exit(1)
    }
  }
}
