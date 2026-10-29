import AVFoundation
import CryptoKit
import Darwin
import Drama3Contracts
import Foundation
import MLX
import Tokenizers

public final class Drama3GenerationSession {
  private let textInput: Drama3TextInput?
  public init(textInput: Drama3TextInput? = nil) { self.textInput = textInput }
  public func run(
    _ validated: Drama3ValidatedRequest, onEvent: @escaping (Drama3Event) async throws -> Void
  ) async throws -> Drama3Receipt {
    var request = validated.request
    let root = URL(fileURLWithPath: request.modelPath)
    let admission = try Drama3ModelLoader.inspect(at: root)
    guard admission.ready, admission.modes.contains(request.mode) else {
      throw Drama3Error.invalid(
        "The checkpoint is incomplete or lacks the control-head supplement for this task: \(admission.missing.joined(separator:", "))."
      )
    }
    let output = URL(fileURLWithPath: request.outputDirectory).standardizedFileURL
    guard !request.outputDirectory.isEmpty, !FileManager.default.fileExists(atPath: output.path)
    else { throw Drama3Error.invalid("Choose a new output take directory.") }
    let start = Date()
    var phaseTimes: [String: Double] = [:]
    var phaseStart = Date()
    let budget =
      request.maxResidentBytes ?? UInt64(Double(ProcessInfo.processInfo.physicalMemory) * 0.85)
    guard
      (request.mode == .infill && request.editOperation == .delete)
        || budget > UInt64(admission.weightBytes) + 2_147_483_648
    else {
      throw Drama3Error.invalid(
        "This memory budget cannot hold the selected checkpoint. Choose Compact weights or increase the explicit budget."
      )
    }
    MLX.Memory.memoryLimit = Int(min(budget, UInt64(Int.max)))
    MLX.Memory.cacheLimit = 256 * 1024 * 1024
    let stage = output.deletingLastPathComponent().appendingPathComponent(
      ".\(output.lastPathComponent)-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
      at: stage.appendingPathComponent("References"), withIntermediateDirectories: true)
    try FileManager.default.createDirectory(
      at: stage.appendingPathComponent("History"), withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: stage) }
    try await onEvent(.init("Checking model and preparing references…", fraction: 0.02))
    let tokenizer = try await AutoTokenizer.from(modelFolder: root)
    guard tokenizer.convertTokenToId("<|semantic:0|>") == 248099,
      tokenizer.convertTokenToId("<|semantic:4095|>") == 252194,
      tokenizer.convertTokenToId("<|im_end|>") == 248046
    else { throw Drama3Error.invalid("Tokenizer and Drama 3 checkpoint disagree.") }
    let cacheRoot = URL(
      fileURLWithPath: request.cacheDirectory.isEmpty
        ? output.deletingLastPathComponent().appendingPathComponent("ReferenceCache").path
        : request.cacheDirectory)
    try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
    let codecFile = root.appendingPathComponent("codec.safetensors")
    let codecInfo = try codecFile.resourceValues(forKeys: [
      .fileSizeKey, .contentModificationDateKey, .fileResourceIdentifierKey,
    ])
    let codecStamp = Self.hash(
      Data(
        "\(codecFile.resolvingSymlinksInPath().path):\(String(describing:codecInfo.fileResourceIdentifier)):\(codecInfo.fileSize ?? 0):\(codecInfo.contentModificationDate?.timeIntervalSince1970 ?? 0)"
          .utf8))
    let fingerprintFile = cacheRoot.appendingPathComponent("codec-" + codecStamp + ".sha256")
    let codecID: String
    if let cached = try? String(contentsOf: fingerprintFile, encoding: .utf8), cached.count == 64 {
      codecID = cached
    } else {
      codecID = try Drama3AudioCodec.digest(codecFile)
      try Data(codecID.utf8).write(to: fingerprintFile, options: .atomic)
    }
    if !request.history.isEmpty || request.sourceCodes != nil {
      guard request.contextCodecFingerprint == codecID else {
        throw Drama3Error.invalid(
          "The prior take uses a different or unverified codec. Use its source audio for an imported edit."
        )
      }
    }
    var codec: Drama3AudioCodec?
    var hits = 0
    var references: [(String, Drama3Codes)] = []
    for i in request.references.indices {
      try Task.checkCancellation()
      let ref = request.references[i]
      if let expected = ref.audioSHA256 {
        guard try Drama3AudioPreparation.fileDigest(ref.audioPath) == expected else {
          throw Drama3Error.invalid(
            "A retained voice sample changed. Restore it before generating with this saved voice.")
        }
      }
      let prepared = try await Drama3AudioPreparation.prepare(
        source: URL(fileURLWithPath: ref.audioPath), start: ref.start, duration: ref.duration,
        channel: ref.channel, in: stage.appendingPathComponent("References"))
      let key = Self.hash(Data((prepared.sha256 + codecID).utf8))
      let cached = cacheRoot.appendingPathComponent(key + ".safetensors")
      let codes: MLXArray
      if let existing = try? Drama3AudioCodec.loadCodes(cached) {
        codes = existing
        hits += 1
      } else {
        if codec == nil { codec = try Drama3AudioCodec(at: root) }
        codes = try codec!.encode(Drama3AudioCodec.readPCM(URL(fileURLWithPath: prepared.path)))
        try Drama3AudioCodec.saveCodes(codes, to: cached)
      }
      let codeFile = stage.appendingPathComponent("References/ref-\(i).safetensors")
      try Drama3AudioCodec.saveCodes(codes, to: codeFile)
      let record = Drama3Codes(
        path: codeFile.path, frames: codes.dim(1), sha256: try Drama3AudioCodec.digest(codeFile))
      references.append((Drama3Text.tagged(ref.text, speaker: ref.speaker), record))
      request.references[i].audioSHA256 = prepared.sha256
      request.references[i].audioPath = prepared.path
      request.references[i].start = 0
      request.references[i].duration = prepared.duration
      request.references[i].channel = "mix"
      request.references[i].codes = record
    }
    for i in request.history.indices {
      let source = URL(fileURLWithPath: request.history[i].codes.path)
      let codes = try Self.read(request.history[i].codes)
      let dest = stage.appendingPathComponent("History/turn-\(i).safetensors")
      if source.standardizedFileURL != dest.standardizedFileURL {
        try Drama3AudioCodec.saveCodes(codes, to: dest)
      }
      request.history[i].codes = .init(
        path: dest.path, frames: codes.dim(1), sha256: try Drama3AudioCodec.digest(dest))
    }
    var infillPrefix: MLXArray?
    var infillSuffix: MLXArray?
    if request.mode == .infill {
      let source: MLXArray
      if let codes = request.sourceCodes {
        source = try Self.read(codes)
        let owned = stage.appendingPathComponent("History/source.safetensors")
        try Drama3AudioCodec.saveCodes(source, to: owned)
        request.sourceCodes = .init(
          path: owned.path, frames: source.dim(1), sha256: try Drama3AudioCodec.digest(owned))
        request.sourceAudio = nil
      } else if let audio = request.sourceAudio {
        let asset = AVURLAsset(url: URL(fileURLWithPath: audio))
        let duration = try await asset.load(.duration).seconds
        let prepared = try await Drama3AudioPreparation.prepare(
          source: URL(fileURLWithPath: audio), start: 0, duration: duration,
          in: stage.appendingPathComponent("References"))
        if codec == nil { codec = try Drama3AudioCodec(at: root) }
        request.sourceAudio = prepared.path
        source = try codec!.encode(Drama3AudioCodec.readPCM(URL(fileURLWithPath: prepared.path)))
      } else if let a = request.prefix, let b = request.suffix {
        infillPrefix = try Self.read(a)
        infillSuffix = try Self.read(b)
        source = concatenated([infillPrefix!, infillSuffix!], axis: 1)
      } else {
        throw Drama3Error.invalid("Choose source speech for infilling.")
      }
      if infillPrefix == nil {
        let p = try Drama3AudioEdit.partition(
          source, start: request.editStart!, end: request.editEnd!)
        infillPrefix = p.prefix
        infillSuffix = p.suffix
      }
      func span(_ codes: MLXArray, _ name: String) throws -> Drama3Codes {
        guard codes.dim(1) > 0 else { return .init(path: "", frames: 0) }
        let url = stage.appendingPathComponent("History/\(name).safetensors")
        try Drama3AudioCodec.saveCodes(codes, to: url)
        return .init(path: url.path, frames: codes.dim(1))
      }
      request.prefix = try span(infillPrefix!, "prefix")
      request.suffix = try span(infillSuffix!, "suffix")
    }
    codec = nil
    MLX.Memory.clearCache()
    phaseTimes["reference_seconds"] = Date().timeIntervalSince(phaseStart)
    phaseStart = Date()
    var warnings: [String] = []
    var allCodes: [MLXArray] = []
    var boundaries: [Int] = []
    var total = 0
    var truncated = false
    var previewIndex = 0
    let previewRoot = output.deletingLastPathComponent().appendingPathComponent(
      output.lastPathComponent + "-preview")
    var completed = false
    defer { if !completed { try? FileManager.default.removeItem(at: previewRoot) } }
    if request.mode != .infill || request.editOperation != .delete {
      let chunks =
        try request.streamingChunks
        ?? (request.mode == .lineContinuation
          ? request.text.components(separatedBy: "\n").filter {
            !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
          }.flatMap { try Drama3Text.chunks($0) }
          : request.mode == .infill ? [request.text] : Drama3Text.chunks(request.text))
      var recordedChunks: [String] = []
      try await onEvent(.init("Loading quantized Drama 3 backbone and audio tower…", fraction: 0.1))
      var weights: Drama3Weights? = try Drama3Weights(at: root, files: admission.files)
      var backbone: Drama3HybridBackbone? = Drama3HybridBackbone(weights!)
      var tower: Drama3CodebookTower? = Drama3CodebookTower(weights!, backbone: backbone!)
      phaseTimes["model_load_seconds"] = Date().timeIntervalSince(phaseStart)
      phaseStart = Date()
      var pinned: Drama3History?
      var latest: Drama3History?
      if request.stream {
        codec = try Drama3AudioCodec(at: root)
        try FileManager.default.createDirectory(at: previewRoot, withIntermediateDirectories: true)
      }
      var previewStream = codec?.makeStream()
      // Codec constructors consume the global PRNG before replacing parameters.
      // Seed after them so live playback cannot change a take's sampled audio.
      MLXRandom.seed(request.seed)
      var index = 0
      while true {
        let chunk: String
        let nextText: String?
        if let textInput {
          if !(await textInput.ready) {
            weights = nil
            tower = nil
            backbone = nil
            MLX.Memory.clearCache()
            try await onEvent(.init("Waiting for text and one-chunk lookahead · model released…"))
          }
          guard let incoming = try await textInput.next() else { break }
          chunk = incoming.text
          nextText = incoming.lookahead
          if weights == nil {
            weights = try Drama3Weights(at: root, files: admission.files)
            backbone = Drama3HybridBackbone(weights!)
            tower = Drama3CodebookTower(weights!, backbone: backbone!)
            if request.stream && codec == nil {
              codec = try Drama3AudioCodec(at: root)
              previewStream = codec?.makeStream()
            }
          }
        } else {
          guard index < chunks.count else { break }
          chunk = chunks[index]
          nextText = index + 1 < chunks.count ? chunks[index + 1] : nil
        }
        recordedChunks.append(chunk)
        guard total + request.sampling.maxFrames <= 262144 else {
          throw Drama3Error.invalid(
            "This take reached the audio-frame budget. Start a new take to continue.")
        }
        let boundaryStart = boundaries.count
        try Task.checkCancellation()
        var target = chunk
        if let language = request.language { target += " [language: \(language)]" }
        if let duration = request.durationHint {
          target += " [audio tokens: \(Int((duration*44100/2048).rounded()))]"
        }
        var history = request.history
        if let pinned { history.append(pinned) }
        if let latest { history.append(latest) }
        let parts = try Drama3Prompt.build(
          text: target, references: references, history: history, mode: request.mode,
          prefix: request.prefix, suffix: request.suffix, direction: request.direction,
          scene: request.scene, characters: request.characters, nextText: nextText)
        let rows = try Self.frames(parts, tokenizer: tokenizer)
        guard rows.dim(0) <= request.maxPromptRows,
          rows.dim(0) + request.sampling.maxFrames <= 262144
        else { throw Drama3Error.invalid("The exact tokenized prompt exceeds its context budget.") }
        let state = Drama3State()
        var hidden: MLXArray?
        let count = rows.dim(0)
        for begin in stride(from: 0, to: count, by: 256) {
          try Task.checkCancellation()
          let end = min(count, begin + 256)
          let frames = rows[begin..<end, 0...].expandedDimensions(axis: 0)
          let h = try backbone!.forward(frames, state: state)
          let labelled = min(end, count - 1) - begin
          if labelled > 0 {
            try tower!.prefill(
              hidden: h[0..., ..<labelled, 0...],
              frames: rows[begin..<(begin + labelled), 0...].expandedDimensions(axis: 0),
              next: rows[(begin + 1)..<(begin + labelled + 1), 0...].expandedDimensions(axis: 0),
              state: state)
          }
          state.evaluate([h])
          hidden = h[0..., (end - begin - 1)..., 0...]
        }
        try await onEvent(
          .init(
            "Generating segment \(index+1)…",
            fraction: min(0.8, 0.15 + 0.65 * Double(index) / Double(chunks.count))))
        var previous = rows[(count - 1)..., 0...].expandedDimensions(axis: 0)
        var generated: [MLXArray] = []
        var seen: [Int32] = []
        var ended = false
        var emitted = 0
        for step in 0..<request.sampling.maxFrames {
          try Task.checkCancellation()
          let logits = try weights!.linear("semantic_head", hidden!)
          var candidates = MLXArray((248099...252194).map(Int32.init) + [248046])
          var distribution = logits
          if request.mode == .scene || request.mode == .lineSeparated {
            let extra = try weights!.linear("control_head", hidden!)
            distribution = concatenated([logits, extra], axis: -1)
            guard let separator = tokenizer.convertTokenToId("<|audio_separator|>") else {
              throw Drama3Error.invalid("Tokenizer lacks the audio separator.")
            }
            candidates = concatenated([candidates, MLXArray([Int32(separator)])])
          }
          let choice = try Drama3Sampler.sample(
            logits: distribution, temperature: request.sampling.temperature,
            topK: request.sampling.topK, topP: request.sampling.topP)
          var main = candidates[choice].reshaped(1, 1)
          let selected = main.item(Int32.self)
          if seen.contains(selected) && (248099...252194).contains(Int(selected))
            && request.sampling.temperature > 0
          {
            main = candidates[
              try Drama3Sampler.sample(
                logits: distribution, temperature: 1, topK: 30, topP: 0.95, checked: false)
            ].reshaped(1, 1)
          }
          let token = main.item(Int32.self)
          if token == 248046 {
            ended = true
            break
          }
          var frame = try tower!.generate(
            hidden: hidden!, main: main, previous: previous, state: state,
            sampling: request.sampling)
          if (248099...252194).contains(Int(token)) {
            let c = frame[0, 0, 1...]
            generated.append(c)
            seen.append(token)
            if seen.count > 16 { seen.removeFirst() }
          } else {
            let boundary = total + generated.count
            if boundary > (boundaries.last ?? 0) {
              boundaries.append(boundary)
            } else {
              warnings.append("An empty scene segment was skipped.")
            }
            frame = concatenated(
              [main[.ellipsis, .newAxis], MLXArray.zeros([1, 1, 10], dtype: .int32)], axis: -1)
          }
          hidden = try backbone!.forward(frame, state: state)
          previous = frame
          state.evaluate([hidden!] + (generated.last.map { [$0] } ?? []))
          if request.stream, let previewStream,
            generated.count - emitted
              >= (emitted == 0 ? request.firstChunkFrames : request.streamIntervalFrames)
          {
            let window = stacked(Array(generated[emitted...]), axis: 1)
            let samples = try previewStream.decode(window)
            let preview = previewRoot.appendingPathComponent("\(previewIndex).wav")
            previewIndex += 1
            do {
              let writer = try AVAudioFile(
                forWriting: preview,
                settings: AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!.settings)
              try Drama3AudioCodec.appendPCM(samples, to: writer)
            }
            emitted = generated.count
            try await onEvent(
              .init("Streaming segment \(index+1)…", audioPreviewPath: preview.path))
          }
          if step % 32 == 0 {
            try await onEvent(
              .init(
                "Segment \(index+1) · \(generated.count) audio frames",
                fraction: min(0.8, 0.15 + 0.65 * Double(index) / Double(chunks.count))))
          }
        }
        if request.stream, let previewStream, generated.count > emitted {
          let window = stacked(Array(generated[emitted...]), axis: 1)
          let samples = try previewStream.decode(window)
          let preview = previewRoot.appendingPathComponent("\(previewIndex).wav")
          previewIndex += 1
          do {
            let writer = try AVAudioFile(
              forWriting: preview,
              settings: AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!.settings)
            try Drama3AudioCodec.appendPCM(samples, to: writer)
          }
          try await onEvent(.init("Streaming final audio…", audioPreviewPath: preview.path))
        }
        if !ended {
          truncated = true
          warnings.append("Segment \(index+1) reached its audio-token budget.")
        }
        guard !generated.isEmpty else {
          throw Drama3Error.invalid("Drama 3 returned an empty speech segment.")
        }
        let codes = stacked(generated, axis: 1).asType(.int32)
        eval(codes)
        let chunkFile = stage.appendingPathComponent("History/chunk-\(index).safetensors")
        try Drama3AudioCodec.saveCodes(codes, to: chunkFile)
        let historyTurn = Drama3History(
          text: target, codes: .init(path: chunkFile.path, frames: codes.dim(1)), nextText: nextText
        )
        if pinned == nil { pinned = historyTurn } else { latest = historyTurn }
        // Materialize compact codes on CPU; do not retain sampled model computation graphs.
        allCodes.append(MLXArray(codes.asArray(Int32.self)).reshaped(10, codes.dim(1)))
        total += codes.dim(1)
        if boundaries.last != total { boundaries.append(total) }
        if request.mode == .scene || request.mode == .lineSeparated {
          let expected = chunk.components(separatedBy: "\n").filter {
            !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
          }.count
          if boundaries.count - boundaryStart != expected {
            truncated = true
            warnings.append(
              "Segment \(index+1) produced \(boundaries.count-boundaryStart) audio lines for \(expected) script lines; audition for skipped or repeated lines."
            )
          }
        }
        index += 1
      }
      phaseTimes["generation_seconds"] = Date().timeIntervalSince(phaseStart)
      phaseStart = Date()
      weights = nil
      tower = nil
      backbone = nil
      previewStream = nil
      codec = nil
      MLX.Memory.clearCache()
      guard !allCodes.isEmpty else {
        throw Drama3Error.invalid("The text stream contained no speech.")
      }
      if request.mode == .streamingText {
        request.streamingChunks = recordedChunks
        request.text = recordedChunks.joined(separator: "\n")
      }
    }
    var codes =
      allCodes.isEmpty ? MLXArray.zeros([10, 0], dtype: .int32) : concatenated(allCodes, axis: 1)
    if let a = infillPrefix, let b = infillSuffix {
      codes = concatenated([a, codes, b], axis: 1)
      boundaries = [codes.dim(1)]
    }
    try Drama3AudioCodec.saveCodes(codes, to: stage.appendingPathComponent("codes.safetensors"))
    try await onEvent(.init("Decoding speech · generator weights released…", fraction: 0.85))
    codec = try Drama3AudioCodec(at: root)
    let audioURL = stage.appendingPathComponent("speech.wav")
    do {
      let writer = try AVAudioFile(
        forWriting: audioURL,
        settings: AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 1)!.settings)
      let decoder = codec!.makeStream()
      for begin in stride(from: 0, to: codes.dim(1), by: 256) {
        try Task.checkCancellation()
        let end = min(codes.dim(1), begin + 256)
        let pcm = try decoder.decode(codes[0..., begin..<end])
        try Drama3AudioCodec.appendPCM(pcm, to: writer)
      }
    }
    codec = nil
    MLX.Memory.clearCache()
    request.mapMediaPaths { value in
      value.hasPrefix(stage.path + "/") ? String(value.dropFirst(stage.path.count + 1)) : value
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(request).write(
      to: stage.appendingPathComponent("request.json"), options: .atomic)
    var usage = rusage()
    getrusage(RUSAGE_SELF, &usage)
    let receipt = Drama3Receipt(
      id: request.id, frames: codes.dim(1) * 2048, codeFrames: codes.dim(1), truncated: truncated,
      audioSHA256: try Drama3AudioCodec.digest(audioURL),
      codesSHA256: try Drama3AudioCodec.digest(stage.appendingPathComponent("codes.safetensors")),
      modelFingerprint: admission.fingerprint, segmentFrames: boundaries,
      elapsedSeconds: Date().timeIntervalSince(start), peakResidentBytes: UInt64(usage.ru_maxrss),
      codecFingerprint: codecID, warnings: warnings.isEmpty ? nil : warnings,
      requestSHA256: try Drama3AudioPreparation.fileDigest(
        stage.appendingPathComponent("request.json").path))
    try encoder.encode(receipt).write(
      to: stage.appendingPathComponent("receipt.json"), options: .atomic)
    phaseTimes["decode_seconds"] = Date().timeIntervalSince(phaseStart)
    try JSONSerialization.data(
      withJSONObject: [
        "phase_seconds": phaseTimes, "reference_cache_hits": hits,
        "reference_count": request.references.count, "preview_chunks": previewIndex,
        "elapsed_seconds": receipt.elapsedSeconds, "peak_resident_bytes": receipt.peakResidentBytes,
      ], options: .sortedKeys
    ).write(to: stage.appendingPathComponent("metrics.json"))
    try Task.checkCancellation()
    try FileManager.default.moveItem(at: stage, to: output)
    completed = true
    return try Drama3TakeBundle.verify(at: output)
  }
  private static func hash(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }
  private static func read(_ codes: Drama3Codes) throws -> MLXArray {
    if let hash = codes.sha256 {
      guard try Drama3AudioCodec.digest(URL(fileURLWithPath: codes.path)) == hash else {
        throw Drama3Error.invalid("Prior take codes changed.")
      }
    }
    let result = try Drama3AudioCodec.loadCodes(URL(fileURLWithPath: codes.path))
    guard result.dim(1) == codes.frames else {
      throw Drama3Error.invalid("Prior take frame count changed.")
    }
    return result
  }
  private static func frames(_ parts: [Drama3PromptPart], tokenizer: Tokenizer) throws -> MLXArray {
    var rows: [MLXArray] = []
    for part in parts {
      if let text = part.text {
        let ids = tokenizer.encode(text: text, addSpecialTokens: false)
        var values = [Int32](repeating: 0, count: ids.count * 11)
        for (i, id) in ids.enumerated() { values[i * 11] = Int32(id) }
        if !ids.isEmpty { rows.append(MLXArray(values).reshaped(ids.count, 11)) }
      }
      if let c = part.codes, c.frames > 0 {
        let codes = try read(c)
        rows.append(concatenated([codes[0...0, 0...] + 248099, codes], axis: 0).transposed(1, 0))
      }
    }
    return concatenated(rows, axis: 0)
  }
}
