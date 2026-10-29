import Drama3Contracts
import Foundation

public enum VoiceSampleArchive {
  public static func archive(_ reference: VoiceReference, in root: URL) async throws
    -> VoiceReference
  {
    try reference.validate(transcriptRequired: true)
    if !reference.sourceHash.isEmpty,
      try Drama3AudioPreparation.fileDigest(reference.path) != reference.sourceHash
    {
      throw StudioError.invalid(
        "The retained voice sample changed. Restore its original audio before importing or saving it."
      )
    }
    let sample = try await Drama3AudioPreparation.prepare(
      source: URL(fileURLWithPath: reference.path), start: reference.start,
      duration: reference.duration, channel: reference.channel, in: root)
    var copy = reference
    copy.assetID = nil
    copy.path = sample.path
    copy.start = 0
    copy.duration = sample.duration
    copy.channel = "mix"
    copy.sourceHash = sample.sha256
    return copy
  }
}
