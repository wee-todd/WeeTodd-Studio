import Foundation
import XCTest
@testable import WeeToddStudio
import StudioCore
import ImageLayout
import ImageIO
import CoreGraphics
import CryptoKit

final class MOSSLayoutAnalysisRoutingTests: XCTestCase {
  @MainActor func testApplyRequiresCapturedDocumentAndNeverStartsGeneration() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    var draft = DrawThingsImageDraft(destination: .init(scope: .project, projectID: store.project.id))
    draft.selectProvider(.nativeMOSS); draft.prompt = "Previous user prompt"
    store.imageDraft = draft
    let id = UUID(), sourceSHA = String(repeating: "a", count: 64)
    let guide = directory.appendingPathComponent("source.png")
    let context = try XCTUnwrap(CGContext(data: nil, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 0,
      space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
    let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(guide as CFURL, "public.png" as CFString, 1, nil))
    CGImageDestinationAddImage(destination, try XCTUnwrap(context.makeImage()), nil); XCTAssertTrue(CGImageDestinationFinalize(destination))
    let size = try ImageLayoutAnalysisGeometry.targetSize(sourceWidth: 1, sourceHeight: 1)
    var layout = ImageLayoutDraft(width: size.width, height: size.height, caption: "Extracted composition")
    layout.guideImagePath = guide.path
    layout.analysis = .init(requestID: id.uuidString, sourceSHA256: sourceSHA, sourceWidth: 1, sourceHeight: 1,
      sourceName: "source.png", modelIdentity: String(repeating: "b", count: 64), recipe: "moss-image-layout-vision-ocr-v1", warnings: [])
    let response: [String: Any] = ["schema": "weetodd-moss-layout-result-v1", "requestID": id.uuidString,
      "sourceSHA256": sourceSHA, "sourceWidth": 1, "sourceHeight": 1, "snapshotWidth": 1, "snapshotHeight": 1,
      "snapshotSHA256": SHA256.hash(data: try Data(contentsOf: guide)).map { String(format: "%02x", $0) }.joined(),
      "modelIdentity": String(repeating: "b", count: 64), "modelDirectory": "/models/qwen",
      "recipe": "moss-image-layout-vision-ocr-v1", "layout": try JSONSerialization.jsonObject(with: JSONEncoder().encode(layout)),
      "effectivePrompt": try MOSSLayoutCompiler.compile(layout).effectivePrompt, "promptTokens": 20, "warnings": [],
      "releasedStages": ["analysis"], "normalizedRequest": ["requestID": id.uuidString, "sourceSHA256": sourceSHA,
        "modelDirectory": "/models/qwen", "targetPixels": 2_000_000]]
    let proposal = try MOSSLayoutAnalysisProposal(result: response, output: directory, requestID: id, sourceSHA256: sourceSHA, modelDirectory: "/models/qwen")
    let captured = CapturedMOSSLayoutAnalysis(proposal: proposal, draft: draft, sessionID: store.documentSessionID)
    try store.applyMOSSLayoutAnalysis(captured)
    XCTAssertEqual(store.imageDraft?.prompt, "Extracted composition")
    XCTAssertTrue(store.project.assets.isEmpty); XCTAssertFalse(store.bridge.busy)
    store.newProject(); store.imageDraft = draft
    XCTAssertThrowsError(try store.applyMOSSLayoutAnalysis(captured))
    XCTAssertEqual(store.imageDraft?.prompt, "Previous user prompt")
  }
  @MainActor func testLayoutAnalysisRunsNativeWhenPythonIsUnavailable() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let worker = directory.appendingPathComponent("worker.sh"), capture = directory.appendingPathComponent("args")
    let text = "#!/bin/sh\n/usr/bin/printf '%s\\n' \"$@\" > '\(capture.path)'\n/usr/bin/printf '%s\\n' '{\"status\":\"ok\",\"result\":{\"schema\":\"weetodd-moss-layout-result-v1\"}}'\n"
    try Data(text.utf8).write(to: worker); try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: worker.path)
    let recipe = directory.appendingPathComponent("request.json")
    try Data(#"{"schema":"weetodd-moss-layout-request-v1"}"#.utf8).write(to: recipe)
    var runtime = RuntimeSettings(root: "/no-repository", pythonPath: "/no-python", profilesDirectory: directory.path)
    runtime.mossWorkerPath = worker.path
    let result = try await Bridge().invoke("moss-layout-analyze", runtime: runtime,
      payload: ["recipePath": recipe.path], output: directory.appendingPathComponent("result"))
    XCTAssertEqual(result["schema"] as? String, "weetodd-moss-layout-result-v1")
    XCTAssertTrue(try String(contentsOf: capture, encoding: .utf8).hasPrefix("analyze-layout\n--request\n"))
  }
}
