import XCTest
@testable import StudioCore

final class PromptAssistantTests: XCTestCase {
  func testInstalledVisionModelFollowsChangedInstructions() throws {
    let env = ProcessInfo.processInfo.environment
    guard let helper = env["WEETODD_TEST_QWEN_HELPER"], let model = env["WEETODD_TEST_QWEN_MODEL"],
      let image = env["WEETODD_TEST_QWEN_IMAGE"] else { throw XCTSkip("Optional installed Qwen3.5 vision qualification") }
    var draft = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .global, projectID: UUID()))
    draft.prompt = "A lone armored warrior stands amidst the skeletal arches of a ruined cathedral, bathed in the warm, flickering glow of an orange fire. The composition follows the rule of thirds, with the warrior positioned slightly off-center against the dark, crumbling stone background. Cinematic lighting and color grading create deep shadows and high contrast, emphasizing the gritty, atmospheric texture of the scene. The mood is intense and dramatic, evoking the feel of a high-stakes cinematic fantasy movie."
    let context = PromptAssistantContext(projectID: draft.destination.projectID, image: draft)
    var outputs = [String]()
    for lighting in ["blue moonlight", "golden sunlight"] {
      let payload: [String: Any] = ["requestID": UUID().uuidString, "modelPath": model,
        "systemPrompt": context.systemPrompt(imageCount: 1),
        "prompt": context.userPrompt(instructions: "Rewrite as exactly one short sentence. Change the orange firelight to \(lighting)."),
        "maxTokens": 256, "images": [["path": image, "label": "Canvas"]]]
      let process = Process(); process.executableURL = URL(fileURLWithPath: helper); process.arguments = ["text"]
      let input = Pipe(); let output = Pipe(); process.standardInput = input; process.standardOutput = output
      process.standardError = FileHandle.nullDevice
      try process.run(); try input.fileHandleForWriting.write(contentsOf: JSONSerialization.data(withJSONObject: payload))
      try input.fileHandleForWriting.close()
      let data = output.fileHandleForReading.readDataToEndOfFile(); process.waitUntilExit()
      XCTAssertEqual(process.terminationStatus, 0)
      let line = try XCTUnwrap(String(data: data, encoding: .utf8)?.split(separator: "\n").last)
      let event = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any])
      let result = try XCTUnwrap(event["value"] as? [String: Any])
      let text = try XCTUnwrap(result["text"] as? String)
      XCTAssertEqual(result["imagesUsed"] as? Int, 1)
      XCTAssertTrue(text.lowercased().contains(lighting), text)
      outputs.append(text)
      print("Vision instruction check (\(lighting)): \(text)")
    }
    XCTAssertNotEqual(outputs[0], outputs[1])
  }

  func testRegenerationUsesLatestInstructionsAndKeepsOriginalAsSource() {
    var clip = Clip(); clip.prompt = "A warrior in a cathedral."
    let context = PromptAssistantContext(project: StudioProject(), clip: clip)
    let first = context.userPrompt(instructions: "Change the light to blue.")
    let second = context.userPrompt(instructions: "Move the scene to a forest.")
    XCTAssertTrue(first.contains("Change the light to blue."))
    XCTAssertFalse(second.contains("Change the light to blue."))
    XCTAssertTrue(second.contains("Move the scene to a forest."))
    XCTAssertTrue(second.contains(clip.prompt))
    XCTAssertNotEqual(first, second)
    XCTAssertEqual(context.original, clip.prompt)
  }
  func testRepeatedAssistantResultsAreIdentifiedRatherThanPresentedAsANewDraft() {
    XCTAssertEqual(PromptAssistantContext.repetitionNotice(output: " A fox.\n", original: "A fox.", previous: ""),
      "The model returned the current prompt unchanged. Try a more specific edit or exclude an image that conflicts with your requested change.")
    XCTAssertNotNil(PromptAssistantContext.repetitionNotice(output: "Blue moonlight.", original: "Orange fire.", previous: "Blue moonlight."))
    XCTAssertNil(PromptAssistantContext.repetitionNotice(output: "A snowy forest.", original: "Orange fire.", previous: "Blue moonlight."))
  }

  func testVisionCandidatesKeepEndpointRolesAndIgnoreAudio() {
    var project = StudioProject(); var clip = Clip()
    let first = MediaAsset(name: "Start", kind: .image, path: "/first.png")
    let last = MediaAsset(name: "End", kind: .image, path: "/last.png")
    let audio = MediaAsset(name: "Sound", kind: .audio, path: "/sound.wav")
    project.assets = [first, last, audio]
    clip.attachments = [Attachment(assetID: first.id, role: .first), Attachment(assetID: audio.id, role: .audioDriver), Attachment(assetID: last.id, role: .last)]
    let context = PromptAssistantContext(project: project, clip: clip)
    XCTAssertEqual(context.images.map(\.path), ["/first.png", "/last.png"])
    XCTAssertTrue(context.images[0].label.contains("First frame"))
    XCTAssertTrue(context.images[1].label.contains("Last frame"))
  }
  func testImageCandidatesIncludeEnabledCanvasAndReferencesOnly() {
    var image = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .global, projectID: UUID()))
    image.canvas = ImageWorkspaceInput(path: "/canvas.png")
    image.moodboard = [ImageWorkspaceInput(path: "/ref1.png"), ImageWorkspaceInput(path: "/ref2.png")]
    image.moodboard[1].enabled = false
    let context = PromptAssistantContext(projectID: image.destination.projectID, image: image)
    XCTAssertEqual(context.images.map(\.path), ["/canvas.png", "/ref1.png"])
    XCTAssertFalse(context.systemPrompt(imageCount: 2).contains("No images are supplied"))
    XCTAssertTrue(context.systemPrompt(imageCount: 0).contains("No images are supplied"))
  }
  func testClipTargetRejectsChangedOrRemovedPrompt() throws {
    var project = StudioProject(); var clip = Clip(engine: .h3)
    clip.prompt = "A fox"; clip.soundscape = "Wind"; project.clips = [clip]
    let target = PromptAssistantContext(project: project, clip: clip)
    XCTAssertNoThrow(try target.validate(project: project, image: nil))
    project.clips[0].prompt = "An owl"
    XCTAssertThrowsError(try target.validate(project: project, image: nil))
    project.clips.removeAll()
    XCTAssertThrowsError(try target.validate(project: project, image: nil))
  }
  func testImageTargetCannotOverwriteAnotherWorkspace() throws {
    let project = StudioProject()
    var image = DrawThingsImageDraft(destination: ImageAssetDestination(scope: .project, projectID: project.id))
    let target = PromptAssistantContext(projectID: project.id, image: image)
    XCTAssertNoThrow(try target.validate(project: project, image: image))
    image.prompt = "Edited while writing"
    XCTAssertThrowsError(try target.validate(project: project, image: image))
    XCTAssertThrowsError(try target.validate(project: StudioProject(), image: image))
    XCTAssertThrowsError(try target.validate(project: project, image: nil))
  }
  func testDiscoveryOnlyListsExistingSupportedModelsAndResolvesLinksWithoutCopying() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let model = root.appendingPathComponent("qwen_3.5_4b_i8x.ckpt")
    try Data("fixture".utf8).write(to: model)
    try Data().write(to: root.appendingPathComponent("qwen_3_vl_4b_instruct_q8p.ckpt"))
    XCTAssertEqual(LocalPromptModels.discover(in: root).map(\.path), [model.path])
    let target = root.appendingPathComponent("external-store")
    try FileManager.default.moveItem(at: model, to: target)
    try FileManager.default.createSymbolicLink(at: model, withDestinationURL: target)
    XCTAssertEqual(LocalPromptModels.discover(in: root).map(\.path), [model.path])
    XCTAssertEqual(try Data(contentsOf: target), Data("fixture".utf8))
  }
}
