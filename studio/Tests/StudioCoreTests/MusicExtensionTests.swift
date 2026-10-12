import XCTest
@testable import StudioCore

final class MusicExtensionTests: XCTestCase {
  func testInstrumentalSectionsSurviveRequest() throws {
    var draft = MusicDraft(); draft.modelPath = "/music"; draft.instrumental = true
    draft.arrangement = MusicArrangement(mode: .ordered, sections: [MusicSection(name: "intro"), MusicSection(name: "outro")])
    XCTAssertEqual(try draft.request().lyrics, "[intro]\n[outro]")
    draft.arrangement?.mode = .timed
    XCTAssertThrowsError(try draft.request())
    draft.arrangement?.sections[0].start = 0; draft.arrangement?.sections[0].end = 15
    draft.arrangement?.sections[1].start = 15; draft.arrangement?.sections[1].end = 30
    XCTAssertEqual(try draft.request().lyrics, "[intro 0:00-0:15]\n[outro 0:15-0:30]")
    draft.arrangement?.sections[1].start = 12
    XCTAssertThrowsError(try draft.request())
  }
  func testNamedRuntimeModelsResolveWithoutSavingPathsInDraft() throws {
    let model = InstalledMusicModel(kind: .generator, name: "YuE2 8-bit", path: "/models/yue")
    let adapter = InstalledMusicModel(kind: .instrumental, name: "Instrumental", path: "/models/inst.safetensors")
    var library = MusicModelSettings(); library.models = [model, adapter]
    var draft = MusicDraft(); draft.modelID = model.id; draft.adapterID = adapter.id; draft.instrumental = true
    let request = try library.request(for: draft)
    XCTAssertEqual(request.modelPath, model.path)
    XCTAssertEqual(request.arLoraPath, adapter.path)
    XCTAssertEqual(request.arLoraStrength, 1)
    let data = try JSONEncoder().encode(request)
    let wire = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    XCTAssertEqual(wire["ar_lora_path"] as? String, adapter.path)
    XCTAssertEqual(wire["ar_lora_strength"] as? Double, 1)
    XCTAssertEqual(try JSONDecoder().decode(MusicRequest.self, from: data).arLoraPath, adapter.path)
    XCTAssertEqual(draft.modelPath, "")
    library.models.removeLast()
    XCTAssertThrowsError(try library.request(for: draft))
  }
  func testLegacyDraftDecodesAndInstrumentalDefaultsToAutomatic() throws {
    let original = MusicDraft()
    var object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(original)) as! [String: Any]
    for key in ["arrangement", "modelID", "adapterID", "adapterStrength", "reference"] { object.removeValue(forKey: key) }
    var restored = try JSONDecoder().decode(MusicDraft.self, from: JSONSerialization.data(withJSONObject: object))
    restored.modelPath = "/legacy"; restored.instrumental = true
    XCTAssertEqual(try restored.request().lyrics, "[instrumental]")
  }
  func testMissingSelectedModelDoesNotSilentlyUseDifferentDefault() {
    var library = MusicModelSettings()
    library.models = [InstalledMusicModel(kind: .generator, name: "Available", path: "/other")]
    var draft = MusicDraft(); draft.modelID = UUID()
    XCTAssertThrowsError(try library.request(for: draft))
  }
  func testReferenceMediaRelocatesButSharedWeightsStayInPlace() throws {
    var project = StudioProject(); var draft = MusicDraft(); draft.modelPath = "/shared/model"
    draft.reference = MusicReference(path: "Media/song.wav")
    draft.reference?.analysis = MusicReferenceAnalysis(abc: "X:1", artifacts: "Media/analysis", sourceSHA256: "hash", sourceIn: 0, duration: 30, melodyOnly: true, eventCount: 1)
    project.musicDraft = draft
    ProjectStorage.mapPaths(&project) { $0.hasPrefix("Media/") ? "/new/" + $0 : $0 }
    XCTAssertEqual(project.musicDraft?.reference?.path, "/new/Media/song.wav")
    XCTAssertEqual(project.musicDraft?.reference?.analysis?.artifacts, "/new/Media/analysis")
    XCTAssertEqual(project.musicDraft?.modelPath, "/shared/model")
  }

}
