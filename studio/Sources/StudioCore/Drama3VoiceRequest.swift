import Drama3Contracts
import Foundation

extension VoiceDraft {
  public func drama3Request() throws -> Drama3Request {
    guard engine == .fishDrama3, precision == "auto", seed >= 0, seed < 4_294_967_296 else {
      throw StudioError.invalid(
        "Choose native Drama 3 and automatic checkpoint precision with a 32-bit seed.")
    }
    var result = Drama3Request(modelPath: modelPath, text: text)
    if let mode = dramaMode {
      guard let value = Drama3Mode(rawValue: mode) else {
        throw StudioError.invalid("Choose a supported Drama 3 task.")
      }
      result.mode = value
    }
    result.maxPromptRows = dramaPromptRows ?? 8192
    result.firstChunkFrames = dramaFirstChunkFrames ?? 4
    result.streamIntervalFrames = dramaStreamIntervalFrames ?? 7
    if let memory = dramaMemoryGiB {
      guard memory.isFinite, memory > 0, memory <= 4096 else {
        throw StudioError.invalid(
          "Use a native memory budget above zero and up to 4,096 GiB, or Automatic.")
      }
      result.maxResidentBytes = UInt64(memory * 1_073_741_824)
    }
    result.seed = UInt64(seed)
    result.direction = dramaDirection ?? ""
    result.scene = dramaScene ?? ""
    result.language = dramaLanguage
    result.durationHint = dramaDurationHint
    result.stream = dramaStream ?? false
    result.sampling.temperature = Float(sampling.temperature)
    result.sampling.topP = Float(sampling.topP)
    result.sampling.topK = sampling.topK
    result.sampling.maxFrames = sampling.maxTokens
    func converted(_ sample: VoiceReference, speaker: Int) throws -> Drama3Reference {
      try sample.validate(transcriptRequired: true)
      guard !sample.transcript.contains("<|") else {
        throw StudioError.invalid(
          "Reference transcripts contain spoken words only; assign speakers in the interface.")
      }
      var ref = Drama3Reference(
        text: sample.transcript, audioPath: sample.path, start: sample.start,
        duration: sample.duration, speaker: speaker)
      ref.channel = sample.channel
      ref.audioSHA256 = sample.sourceHash.isEmpty ? nil : sample.sourceHash
      return ref
    }
    if usesDialogue == true {
      guard let dialogue, !dialogue.turns.isEmpty, dialogue.turns.count <= 64,
        Set(dialogue.turns.map(\.id)).count == dialogue.turns.count
      else { throw StudioError.invalid("Add between 1 and 64 scene lines with distinct line IDs.") }
      var mapping = dramaSpeakerMap ?? Drama3SpeakerMap()
      let slots = try mapping.assign(dialogue.speakers.map(\.id))
      for s in dialogue.speakers {
        let slot = slots[s.id]!
        result.characters.append(
          .init(
            id: s.characterID ?? s.id, token: slot, name: s.name,
            description: ([s.voiceProfile?.description ?? ""] + (try s.fishDirection?.tags() ?? []))
              .joined(separator: ", ")))
        guard s.referenceMode != .customVoice && s.referenceMode != .speakerIdentityOnly else {
          throw StudioError.invalid(
            "Drama 3 characters use synthetic voices or audio with transcripts.")
        }
        if s.referenceMode == .audioAndTranscript {
          let refs =
            s.reference.map { [$0] + (s.dramaReferences ?? Array((s.voiceProfile?.samples ?? []).dropFirst())) }
            ?? s.dramaReferences ?? s.voiceProfile?.samples ?? []
          guard !refs.isEmpty else {
            throw StudioError.invalid("Add samples or choose a saved voice for \(s.name).")
          }
          for ref in refs { result.references.append(try converted(ref, speaker: slot)) }
        }
      }
      result.text = try dialogue.turns.map { line in
        guard let speaker = dialogue.speakers.first(where: { $0.id == line.speakerID }),
          let slot = slots[line.speakerID], line.reference == nil
        else {
          throw StudioError.invalid(
            "Assign each line to a character. Native scenes use character samples; use delivery cues for line performance."
          )
        }
        let tags = try (line.fishDirection ?? speaker.fishDirection)?.tags() ?? []
        return "<|speaker:\(slot)|> " + tags.map { "[\($0)] " }.joined() + line.text
      }.joined(separator: "\n")
      if result.mode == .speak { result.mode = .lineContinuation }
    } else {
      guard referenceMode == .synthetic || referenceMode == .audioAndTranscript else {
        throw StudioError.invalid("Drama 3 uses a synthetic voice or audio with a transcript.")
      }
      if referenceMode == .audioAndTranscript {
        let refs = ([reference].compactMap { $0 }) + (dramaReferences ?? [])
        let selected = refs.isEmpty ? (voiceProfile?.samples ?? []) : refs
        guard !selected.isEmpty else {
          throw StudioError.invalid("Choose a saved voice or add cloning samples.")
        }
        result.references = try selected.map { try converted($0, speaker: 0) }
      }
      let tags = try fishDirection?.tags() ?? []
      if !tags.isEmpty { result.text = tags.map { "[\($0)] " }.joined() + result.text }
      if characterID != nil || voiceProfile != nil {
        result.characters = [
          .init(
            id: characterID ?? voiceProfile!.reference.id, token: 0,
            name: voiceProfile?.name ?? name, description: voiceProfile?.description ?? "")
        ]
      }
    }
    if result.mode == .infill { result.references = [] }
    if result.mode == .infill {
      if let operation = dramaEditOperation {
        guard let op = Drama3EditOperation(rawValue: operation) else {
          throw StudioError.invalid("Choose a supported edit operation.")
        }
        result.editOperation = op
      }
      result.sourceAudio = dramaSourceAudio
      result.editStart = dramaEditStart
      result.editEnd = result.editOperation == .insert ? dramaEditStart : dramaEditEnd
    }
    return result
  }
  public func request(resolvedDrama3 native: Drama3Request) throws -> VoiceRequest {
    guard engine == .fishDrama3, native.modelPath == modelPath else {
      throw StudioError.invalid("The native speech model changed during preparation.")
    }
    _ = try native.validate()
    var value = VoiceRequest(
      engine: engine, modelPath: modelPath, precision: "auto", text: native.text,
      referenceMode: referenceMode, reference: reference, language: language, seed: seed,
      sampling: sampling)
    value.drama3 = native
    return value
  }
}
