import Foundation
import ImageLayout
import StudioCore

struct CapturedMOSSLayoutAnalysis {
  let proposal: MOSSLayoutAnalysisProposal
  let draft: DrawThingsImageDraft
  let sessionID: UUID
}

@MainActor extension StudioStore {
  func analyzeMOSSLayout(source: URL, modelDirectory: String) async throws -> CapturedMOSSLayoutAnalysis {
    guard let draft = imageDraft, draft.executionProvider == .nativeMOSS, !draft.hasManagedCharacterPrompt,
      !operationBusy, !preparingImageRequest else { throw StudioError.invalid("Open an idle Local MOSS image draft first.") }
    preparingImageRequest = true; defer { preparingImageRequest = false }
    let session = documentSessionID, id = UUID()
    let shared = draft.mossImage?.sharedDirectory ?? ""
    let countingTokenizer = shared.isEmpty ? nil : URL(fileURLWithPath: shared).appendingPathComponent("tokenizer").path
    let folder = dataDirectory.appendingPathComponent("Cache/MOSS Layout Analysis/" + id.uuidString)
    let prepared = try await Task.detached(priority: .userInitiated) {
      let request = try MOSSLayoutAnalysisPreparation.prepare(source: source, modelDirectory: modelDirectory,
        mossTokenizerDirectory: countingTokenizer, requestID: id)
      try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
      try request.bytes.write(to: folder.appendingPathComponent("request.json"), options: .withoutOverwriting)
      return request
    }.value
    try Task.checkCancellation()
    guard documentSessionID == session, imageDraft == draft else { throw StudioError.invalid("The image draft changed before analysis started.") }
    let response = try await bridge.invoke("moss-layout-analyze", runtime: runtime,
      payload: ["recipePath": folder.appendingPathComponent("request.json").path], output: folder)
    let data = try JSONSerialization.data(withJSONObject: response)
    let proposal = try await Task.detached(priority: .userInitiated) {
      guard let result = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw StudioError.invalid("Invalid image analysis response.")
      }
      return try MOSSLayoutAnalysisProposal(result: result, output: folder, requestID: id,
        sourceSHA256: prepared.sourceSHA256, modelDirectory: modelDirectory, mossTokenizerDirectory: prepared.mossTokenizerDirectory,
        expectedModelFamily: prepared.modelFamily)
    }.value
    try Task.checkCancellation()
    return .init(proposal: proposal, draft: draft, sessionID: session)
  }
  func applyMOSSLayoutAnalysis(_ captured: CapturedMOSSLayoutAnalysis, sourceEnabled: Bool = false, noiseLevel: Double = 0.08) throws {
    guard documentSessionID == captured.sessionID, let current = imageDraft, !operationBusy else {
      throw StudioError.invalid("The document changed. Reopen image analysis for the current draft.")
    }
    imageDraft = try captured.proposal.applying(to: current, captured: captured.draft, sourceEnabled: sourceEnabled, noiseLevel: noiseLevel)
    imageEstimate = nil; preparedMOSSImage = nil; imagePreviewPath = nil
    notice = "Loaded \(captured.proposal.layout.regions.count) regions. Review the layout before generating."
  }
}
