import Darwin
import Drama3Contracts
import Drama3MLX
import Foundation
import MLX

@main struct Drama3Worker {
  static func emit(_ value: [String: Any]) {
    if let data = try? JSONSerialization.data(
      withJSONObject: value, options: [.sortedKeys, .fragmentsAllowed])
    {
      FileHandle.standardOutput.write(data)
      FileHandle.standardOutput.write(Data([10]))
    }
  }
  static func main() async {
    let args = Array(CommandLine.arguments.dropFirst())
    let command = args.first ?? "help"
    func option(_ name: String) -> String? {
      guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
      return args[i + 1]
    }
    if command == "help" || args.contains("--help") {
      print(
        "WeeToddDrama3Worker inspect|preflight|render|stream-text|supplement-head --request FILE --output DIRECTORY\nBuilt with Fish Audio. Native Swift/MLX; no Python inference."
      )
      return
    }
    let library = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
      .deletingLastPathComponent().appendingPathComponent("Resources/Drama3Native/mlx.metallib")
    if FileManager.default.fileExists(atPath: library.path) { GPU.metallib = library }
    let task = Task { () throws -> Void in
      if command == "supplement-head" {
        var setup: [String: Any] = [:]
        if let input = option("--request") {
          let file = URL(fileURLWithPath: input)
          guard (try file.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? Int.max) <= 16 * 1024 * 1024 else {
            throw Drama3Error.invalid("Native request exceeds 16 MiB.")
          }
          setup = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any] ?? [:]
        }
        guard let source = option("--source") ?? setup["source"] as? String,
          let model = option("--model") ?? setup["modelPath"] as? String else {
          throw Drama3Error.invalid(
            "Supply --source original-shard and --model native-model-folder.")
        }
        let sourceURL = URL(fileURLWithPath: source)
        let root = URL(fileURLWithPath: model)
        let ownership = try Drama3SceneSetupLock(model: root)
        defer { withExtendedLifetime(ownership) {} }
        guard
          try Drama3AudioCodec.digest(sourceURL)
            == "c5b065cce3b3b851369dfdb5a3b60f6abf2699121071c2b1a9dfd84acb7b8f7d"
        else {
          throw Drama3Error.invalid(
            "Original source shard checksum differs from the pinned release.")
        }
        let tokenizer =
          try JSONSerialization.jsonObject(
            with: Data(contentsOf: root.appendingPathComponent("tokenizer.json"))) as! [String: Any]
        guard let tokens = tokenizer["added_tokens"] as? [[String: Any]],
          let id = tokens.first(where: { $0["content"] as? String == "<|audio_separator|>" })?["id"]
            as? Int
        else { throw Drama3Error.invalid("Tokenizer lacks the required control token.") }
        try Drama3ControlRows.supplement(
          source: sourceURL, destination: root.appendingPathComponent("control_head.safetensors"),
          tokenIDs: [id])
        emit(["status": "ok", "result": ["control_token_id": id]])
        return
      }
      guard let inputFile = option("--request") else {
        throw Drama3Error.invalid("Supply a native request JSON file.")
      }
      let url = URL(fileURLWithPath: inputFile)
      guard
        (try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? Int.max) <= 16 * 1024 * 1024
      else { throw Drama3Error.invalid("Native request exceeds 16 MiB.") }
      let bytes = try Data(contentsOf: url)
      if command == "inspect" {
        let object = try JSONSerialization.jsonObject(with: bytes) as! [String: Any]
        let admission = try Drama3ModelLoader.inspect(
          at: URL(fileURLWithPath: object["modelPath"] as? String ?? ""))
        emit([
          "status": "ok",
          "result": try JSONSerialization.jsonObject(with: JSONEncoder().encode(admission)),
        ])
        return
      }
      var request = try JSONDecoder().decode(Drama3Request.self, from: bytes)
      request.resolveMediaPaths(relativeTo: url.deletingLastPathComponent())
      if let output = option("--output") { request.outputDirectory = output }
      let validated = try request.validate()
      if command == "preflight" {
        let admission = try Drama3ModelLoader.inspect(at: URL(fileURLWithPath: request.modelPath))
        guard admission.ready, admission.modes.contains(request.mode) else {
          throw Drama3Error.invalid("Complete native model setup for this speech task.")
        }
        emit([
          "status": "ok",
          "result": ["native": true, "ready": true, "model_fingerprint": admission.fingerprint],
        ])
        return
      }
      guard command == "render" || command == "stream-text" else {
        throw Drama3Error.invalid("Unknown native speech command.")
      }
      let input: Drama3TextInput? = command == "stream-text" ? Drama3TextInput() : nil
      if let input {
        guard request.mode == .streamingText else {
          throw Drama3Error.invalid("Live input requires the streamingText task.")
        }
        try await input.append(request.text)
        Task.detached {
          var buffer = Data()
          while true {
            let bytes = FileHandle.standardInput.availableData
            if bytes.isEmpty {
              await input.finish()
              return
            }
            buffer.append(bytes)
            if buffer.count > 65536 {
              await input.cancel()
              return
            }
            while let end = buffer.firstIndex(of: 10) {
              let line = buffer.prefix(upTo: end)
              buffer.removeSubrange(...end)
              var sequence: Int?
              do {
                let message = try JSONSerialization.jsonObject(with: line) as? [String: Any]
                sequence = message?["sequence"] as? Int
                if message?["op"] as? String == "finish" {
                  await input.finish()
                  return
                }
                guard message?["op"] as? String == "append", let text = message?["text"] as? String
                else { throw Drama3Error.invalid("Send an append text or finish message.") }
                try await input.append(text)
                emit([
                  "event": "progress", "message": "Text accepted.", "inputSequence": sequence ?? -1,
                  "inputAccepted": true,
                ])
              } catch {
                emit([
                  "event": "progress", "message": error.localizedDescription,
                  "inputSequence": sequence ?? -1, "inputAccepted": false,
                ])
              }
            }
          }
        }
      }
      let receipt = try await Drama3GenerationSession(textInput: input).run(validated) { event in
        let object =
          try JSONSerialization.jsonObject(with: JSONEncoder().encode(event)) as! [String: Any]
        emit(object)
      }
      let root = URL(fileURLWithPath: request.outputDirectory)
      emit([
        "status": "ok",
        "result": [
          "audio": root.appendingPathComponent(receipt.audio).path, "duration": receipt.duration,
          "sample_rate": receipt.sampleRate, "frames": receipt.frames, "channels": receipt.channels,
          "artifacts": root.path, "truncated": receipt.truncated, "code_frames": receipt.codeFrames,
          "elapsed_seconds": receipt.elapsedSeconds,
          "peak_resident_bytes": receipt.peakResidentBytes,
        ],
      ])
    }
    signal(SIGINT, SIG_IGN)
    signal(SIGTERM, SIG_IGN)
    let interrupted = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
    let terminated = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .global())
    interrupted.setEventHandler { task.cancel() }
    terminated.setEventHandler { task.cancel() }
    interrupted.resume()
    terminated.resume()
    do { try await task.value } catch is CancellationError {
      emit([
        "status": "cancelled", "error": "Speech generation cancelled; native resources released.",
      ])
      exit(130)
    } catch {
      emit(["status": "error", "error": error.localizedDescription])
      exit(1)
    }
  }
}
