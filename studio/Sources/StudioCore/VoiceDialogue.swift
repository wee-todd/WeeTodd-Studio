import Foundation

public struct VoiceSpeaker: Codable, Equatable, Identifiable {
  public var id = UUID()
  public var name: String
  public var referenceMode: VoiceReferenceMode = .audioAndTranscript
  public var reference: VoiceReference?
  public var presetVoice: String?
  public var instructions: String?
  public var fishDirection: FishVoiceDirection?
  public init(name: String, reference: VoiceReference? = nil) { self.name = name; self.reference = reference }
}
public struct VoiceTurn: Codable, Equatable, Identifiable {
  public var id = UUID()
  public var speakerID: UUID
  public var text: String
  public var gapAfter: Double = 0.25
  /// Optional alternate performance by the same speaker (e.g. an emotional sample).
  public var reference: VoiceReference?
  public var instructions: String?
  public var fishDirection: FishVoiceDirection?
  public init(speakerID: UUID, text: String = "") { self.speakerID = speakerID; self.text = text }
}
public struct VoiceDialogue: Codable, Equatable {
  public var speakers: [VoiceSpeaker]
  public var turns: [VoiceTurn]
  public init(speakers: [VoiceSpeaker], turns: [VoiceTurn]) { self.speakers = speakers; self.turns = turns }
  public mutating func assignSpeaker(_ speakerID: UUID, to turnID: UUID) {
    guard speakers.contains(where: { $0.id == speakerID }), let index = turns.firstIndex(where: { $0.id == turnID }),
      turns[index].speakerID != speakerID else { return }
    turns[index].speakerID = speakerID; turns[index].reference = nil
    turns[index].fishDirection = nil
  }
  public mutating func mapPaths(_ transform: (String) throws -> String) rethrows {
    for i in speakers.indices { if let ref = speakers[i].reference { speakers[i].reference?.path = try transform(ref.path) } }
    for i in turns.indices { if let ref = turns[i].reference { turns[i].reference?.path = try transform(ref.path) } }
  }
}
public struct VoiceTurnRequest: Codable, Equatable {
  public var id: UUID
  public var speakerID: UUID
  public var speakerName: String
  public var text: String
  public var referenceMode: VoiceReferenceMode
  public var reference: VoiceReference?
  public var gapAfter: Double
  public var speaker: String?
  public var instruct: String?
  public var voiceDirection: [String]? = nil
  enum CodingKeys: String, CodingKey {
    case id, speakerID = "speaker_id", speakerName = "speaker_name", text
    case referenceMode = "reference_mode", reference, gapAfter = "gap_after", speaker, instruct
    case voiceDirection = "voice_direction"
  }
}
extension VoiceDraft {
  public mutating func startDialogue() {
    if dialogue == nil {
      var speaker = VoiceSpeaker(name: "Speaker 1", reference: reference); speaker.referenceMode = referenceMode
      speaker.presetVoice = presetVoice; speaker.instructions = instructions
      speaker.fishDirection = fishDirection
      dialogue = VoiceDialogue(speakers: [speaker], turns: [VoiceTurn(speakerID: speaker.id, text: text)])
    }
    usesDialogue = true
  }
  public func dialogueRequest() throws -> VoiceRequest {
    guard let dialogue, !dialogue.turns.isEmpty, dialogue.turns.count <= 64,
      Set(dialogue.turns.map(\.id)).count == dialogue.turns.count,
      Set(dialogue.speakers.map(\.id)).count == dialogue.speakers.count else {
      throw StudioError.invalid("Add between 1 and 64 dialogue lines with unique speakers and line identifiers.")
    }
    var first: VoiceRequest?
    var turns: [VoiceTurnRequest] = []
    for turn in dialogue.turns {
      guard let speaker = dialogue.speakers.first(where: { $0.id == turn.speakerID }),
        !speaker.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
        turn.gapAfter.isFinite, (0...10).contains(turn.gapAfter) else {
        throw StudioError.invalid("Assign every line to a named speaker and use pauses from 0 to 10 seconds.")
      }
      var line = self; line.usesDialogue = false; line.dialogue = nil; line.text = turn.text
      line.referenceMode = speaker.referenceMode; line.reference = turn.reference ?? speaker.reference
      line.presetVoice = speaker.presetVoice; line.instructions = turn.instructions ?? speaker.instructions
      line.fishDirection = turn.fishDirection ?? speaker.fishDirection
      let request = try line.request()
      if first == nil { first = request }
      turns.append(VoiceTurnRequest(id: turn.id, speakerID: speaker.id, speakerName: speaker.name,
        text: request.text, referenceMode: request.referenceMode, reference: request.reference, gapAfter: turn.gapAfter, speaker: request.speaker, instruct: request.instruct,
        voiceDirection: request.voiceDirection))
    }
    var result = first!; result.turns = turns
    return result
  }
}
