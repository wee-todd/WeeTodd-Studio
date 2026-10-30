import Foundation

public enum NativeRestorationCLI {
  private static func write<T: Encodable>(_ value: T) throws {
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
    FileHandle.standardOutput.write(try encoder.encode(value) + Data([10]))
  }
  private static func json<T: Encodable>(_ value: T) throws -> Any {
    try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
  }
  private static func completion(_ value: [String: Any]) throws {
    FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]) + Data([10]))
  }
  static func data(_ filename: String, limit: Int = 1024 * 1024) throws -> Data {
    let url = URL(fileURLWithPath: filename).resolvingSymlinksInPath(), before = try NativeMovieFrozenMedia.Identity(url)
    guard before.bytes <= limit else {
      throw StudioError.invalid("A restoration request must be a regular JSON file of at most 1 MiB.")
    }
    let input = try FileHandle(forReadingFrom: url); defer { try? input.close() }
    let bytes = try input.read(upToCount: limit + 1) ?? Data()
    guard bytes.count == before.bytes, before == (try NativeMovieFrozenMedia.Identity(url)) else {
      throw StudioError.invalid("Native restoration metadata changed while reading.")
    }
    return bytes
  }
  public static func run(mode: String, arguments: [String]) async throws {
    guard ["regenerate", "process"].contains(mode) else { throw StudioError.invalid("Unknown native restoration command.") }
    let options = try FlashVSRCLIArguments(arguments), bytes = try data(options.requestPath)
    let output = URL(fileURLWithPath: options.outputDirectory)
    if mode == "regenerate" {
      guard !options.resume else { throw StudioError.invalid("An interrupted H3 stage restarts in a new output directory. Use process for verified completed-stage reuse.") }
      let request = try JSONDecoder().decode(H3RegenerateRequest.self, from: bytes), service = H3RegenerateService()
      let plan = try await service.preflight(request: request)
      if options.preflightOnly { try completion(["status": "success", "preflight": true, "plan": json(plan)]) }
      else {
        let receipt = try await service.run(plan: plan, directory: output, emit: { event in try? write(event) })
        try completion(["status": "success", "result": json(receipt)])
      }
      return
    }
    let request = try JSONDecoder().decode(NativeVideoProcessingRequest.self, from: bytes)
    try request.validate()
    // Source timing precedes component hashing/admission, and is checked again
    // under the sequence cache lease before any weighted model executes.
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("weetodd-process-cli-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    let runtime = request.runtime ?? request.h3Runtime?.mediaRuntime ?? request.nativeRuntime?.mediaRuntime
    guard let runtime else { throw StudioError.invalid("The native processing media runtime is missing.") }
    let source = try await Task.detached { try FlashVSRFile.capture(URL(fileURLWithPath: request.sourcePath)) }.value
    guard request.sourceSHA256 == nil || request.sourceSHA256 == source.sha256 else { throw StudioError.invalid("The frozen processing source changed.") }
    let media = try await Task.detached {
      try FlashVSRMediaInfo.inspect(URL(fileURLWithPath: source.path), runtime: runtime, directory: root, cancellation: .init())
    }.value
    if request.recipe.hasH3Regenerate { try H3RegenerateSettings().validateSourceClock(numerator: media.fps.numerator, denominator: media.fps.denominator) }
    let plan = try await NativeVideoProcessingPlan.preflight(recipe: request.recipe, runtime: request.runtime,
      nativeRuntime: request.nativeRuntime, h3Runtime: request.h3Runtime)
    if options.preflightOnly { try completion(["status": "success", "preflight": true, "plan": json(plan)]) }
    else {
      guard options.resume || !FileManager.default.fileExists(atPath: output.path) else {
        throw StudioError.invalid("Choose a new processing cache or pass --resume to reuse verified completed stages.")
      }
      let result = try await NativeVideoProcessingExecutor.run(plan: plan, source: source, directory: output, emit: { event in try? write(event) })
      let receipt = try JSONSerialization.jsonObject(with: data(result.receiptPath, limit: 8 * 1024 * 1024))
      try completion(["status": "success", "result": receipt, "receiptPath": result.receiptPath, "reusedSteps": result.reusedSteps])
    }
  }
}
