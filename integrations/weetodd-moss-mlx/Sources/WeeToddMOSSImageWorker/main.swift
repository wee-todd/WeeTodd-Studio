import Foundation
import MOSSImageMLX
import Darwin

@main struct Worker {
  static func main() async {
    do {
      let arguments = Array(CommandLine.arguments.dropFirst())
      if arguments.first == "analyze-layout" {
        guard let flag = arguments.firstIndex(of: "--request"), arguments.indices.contains(flag + 1) else {
          throw MOSSError("analyze-layout --request request.json --output proposal-directory")
        }
        let file = try FileHandle(forReadingFrom: URL(fileURLWithPath: arguments[flag + 1]))
        defer { try? file.close() }
        let data = try file.read(upToCount: 2 * 1024 * 1024 + 1) ?? Data()
        guard data.count <= 2 * 1024 * 1024 else { throw MOSSError("Analysis request exceeds 2 MiB.") }
        var request = try JSONDecoder().decode(MOSSLayoutAnalysisRequest.self, from: data)
        if let output = arguments.firstIndex(of: "--output"), arguments.indices.contains(output + 1) {
          request.outputDirectory = arguments[output + 1]
        }
        signal(SIGINT, SIG_IGN)
        let task = Task { [request] in
          try await MOSSLayoutAnalyzer.analyze(request) { stage, message, fraction in
            if let bytes = try? JSONSerialization.data(withJSONObject: ["event": "progress", "stage": stage,
              "message": message, "fraction": fraction], options: .sortedKeys) {
              FileHandle.standardOutput.write(bytes); FileHandle.standardOutput.write(Data([10]))
            }
          }
        }
        let interrupt = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global(qos: .userInitiated))
        interrupt.setEventHandler { @Sendable in task.cancel() }; interrupt.resume(); defer { interrupt.cancel() }
        let result = try JSONSerialization.jsonObject(with: await task.value)
        let bytes = try JSONSerialization.data(withJSONObject: ["status": "ok", "ok": true, "result": result],
          options: [.sortedKeys, .withoutEscapingSlashes])
        FileHandle.standardOutput.write(bytes); FileHandle.standardOutput.write(Data([10])); return
      }
      if arguments.first == "convert-model" {
        func option(_ name: String) -> String? {
          guard let i = arguments.firstIndex(of: name), arguments.indices.contains(i + 1) else { return nil }; return arguments[i + 1]
        }
        guard let input = option("--input"), let output = option("--output"),
          let bits = option("--bits").flatMap(Int.init), let component = option("--component") else {
          throw MOSSError("convert-model --input component-folder --output new-model-library-folder --bits 8|4 --component transformer|text_encoder")
        }
        signal(SIGINT, SIG_IGN)
        let task = Task { () throws -> String in
          try MOSSQuantization.convert(input: URL(fileURLWithPath: input), output: URL(fileURLWithPath: output), bits: bits, component: component) { message, fraction in
            if let bytes = try? JSONSerialization.data(withJSONObject: ["event": "progress", "stage": "conversion", "message": message, "fraction": fraction]) {
              FileHandle.standardOutput.write(bytes); FileHandle.standardOutput.write(Data([10]))
            }
          }
          return output
        }
        let interrupt = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global(qos: .userInitiated))
        interrupt.setEventHandler { @Sendable in task.cancel() }; interrupt.resume(); defer { interrupt.cancel() }
        let destination = try await task.value
        let bytes = try JSONSerialization.data(withJSONObject: ["status": "ok", "ok": true, "result": ["componentDirectory": destination, "bits": bits]])
        FileHandle.standardOutput.write(bytes); FileHandle.standardOutput.write(Data([10])); return
      }
      guard let verb = arguments.first, ["preflight", "inspect-model", "render", "vae-roundtrip"].contains(verb),
        let flag = arguments.firstIndex(of: "--request"), arguments.indices.contains(flag + 1) else {
        throw MOSSError("Usage: WeeToddMOSSImageWorker preflight|render --request request.json [--output take-directory] [--capture directory]")
      }
      let file = URL(fileURLWithPath: arguments[flag + 1])
      let data = try Data(contentsOf: file)
      guard data.count <= 2 * 1024 * 1024 else { throw MOSSError("MOSS request exceeds 2 MiB.") }
      var request = try JSONDecoder().decode(MOSSImageRequest.self, from: data)
      if arguments.contains("--profile") { request.configuration.profile = true }
      if let flag = arguments.firstIndex(of: "--output"), arguments.indices.contains(flag + 1) {
        request.outputPath = URL(fileURLWithPath: arguments[flag + 1]).appendingPathComponent("image.png").path
      }
      let capture: URL?
      if let flag = arguments.firstIndex(of: "--capture"), arguments.indices.contains(flag + 1) { capture = URL(fileURLWithPath: arguments[flag + 1]) } else { capture = nil }
      signal(SIGINT, SIG_IGN)
      let task = Task { [request, capture] in
        let prepared = try await MOSSImagePreflight.inspect(request)
        if verb == "render" || verb == "vae-roundtrip" {
          return try MOSSImageRunner.render(prepared, captureDirectory: capture, vaeRoundtrip: verb == "vae-roundtrip") { stage, message, fraction in
            let event: [String: Any] = ["event": "progress", "stage": stage, "message": message, "fraction": fraction]
            if let bytes = try? JSONSerialization.data(withJSONObject: event, options: .sortedKeys) {
              FileHandle.standardOutput.write(bytes); FileHandle.standardOutput.write(Data([10]))
            }
          }
        }
        return MOSSRunReceipt(prepared)
      }
      let interrupt = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global(qos: .userInitiated))
      // A dispatch callback runs off-main; explicitly Sendable prevents Swift 6
      // from inheriting async main's actor and trapping when SIGINT arrives.
      interrupt.setEventHandler { @Sendable in task.cancel() }; interrupt.resume()
      defer { interrupt.cancel() }
      let receipt = try await task.value
      let result = try JSONSerialization.jsonObject(with: JSONEncoder().encode(receipt))
      let bytes = try JSONSerialization.data(withJSONObject: ["status": "ok", "ok": true, "result": result], options: [.sortedKeys, .withoutEscapingSlashes])
      FileHandle.standardOutput.write(bytes); FileHandle.standardOutput.write(Data([10]))
    } catch {
      let cancelled = error is CancellationError
      let bytes = try! JSONSerialization.data(withJSONObject: ["status": cancelled ? "cancelled" : "error", "ok": false, "error": cancelled ? "Job cancelled." : error.localizedDescription], options: .sortedKeys)
      FileHandle.standardOutput.write(bytes); FileHandle.standardOutput.write(Data([10])); exit(cancelled ? 130 : 1)
    }
  }
}
