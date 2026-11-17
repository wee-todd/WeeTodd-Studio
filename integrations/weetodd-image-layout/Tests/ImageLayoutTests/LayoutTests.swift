import Foundation
import XCTest
@testable import ImageLayout

final class LayoutTests: XCTestCase {
  func testEditableExportRetainsOwnedFaceMovement() throws {
    let subject = ImageLayoutRegion(type: .character, caption: "Person", bounds: .init(x: 0.1, y: 0.2, width: 0.4, height: 0.7))
    var face = ImageLayoutRegion(type: .object, caption: "Face", bounds: .init(x: 0.2, y: 0.25, width: 0.1, height: 0.1))
    face.parentID = subject.id; face.analysisRole = .face
    var background = ImageLayoutRegion(type: .object, caption: "Cream background", bounds: .init(x: 0, y: 0, width: 1, height: 1))
    background.analysisRole = .canvasBackground
    let draft = ImageLayoutDraft(width: 1024, height: 1024, caption: "Portrait", regions: [background, subject, face])
    let reopened = try MOSSLayoutInterchange.importEditableDraft(MOSSLayoutInterchange.exportEditableDraft(draft))
    XCTAssertEqual(reopened, draft)
    XCTAssertEqual(reopened.regions.map(\.id), draft.regions.map(\.id))
    XCTAssertEqual(reopened.regions[2].parentID, subject.id)
    XCTAssertEqual(reopened.regions[2].analysisRole, .face)
    XCTAssertEqual(reopened.regions[0].analysisRole, .canvasBackground)
    var moved = reopened.regions[1]; moved.bounds.x = 0.5
    let edited = try ImageLayoutEditing.replacing(moved, in: reopened)
    XCTAssertEqual(edited.regions[2].bounds.x, 0.6, accuracy: 1e-9)
  }

  func testEditableImportRejectsUnknownVersionDuplicateKeysAndInvalidOwnership() throws {
    var draft = ImageLayoutDraft(width: 1024, height: 1024, caption: "Portrait", regions: [
      .init(type: .character, caption: "Person", bounds: .init(x: 0.1, y: 0.1, width: 0.8, height: 0.8))])
    let valid = try MOSSLayoutInterchange.exportEditableDraft(draft)
    let text = String(decoding: valid, as: UTF8.self)
    XCTAssertThrowsError(try MOSSLayoutInterchange.importEditableDraft(Data(text.replacingOccurrences(of: "weetodd-image-layout-v1", with: "weetodd-image-layout-v99").utf8)))
    let duplicate = "{\"format\":\"weetodd-image-layout-v1\",\"format\":\"weetodd-image-layout-v1\",\"layout\":" + String(decoding: try JSONEncoder().encode(draft), as: UTF8.self) + "}"
    XCTAssertThrowsError(try MOSSLayoutInterchange.importEditableDraft(Data(duplicate.utf8)))
    draft.regions[0].parentID = UUID()
    let invalid = try JSONSerialization.data(withJSONObject: ["format": "weetodd-image-layout-v1",
      "layout": JSONSerialization.jsonObject(with: JSONEncoder().encode(draft))])
    XCTAssertThrowsError(try MOSSLayoutInterchange.importEditableDraft(invalid))
  }

  func testPortraitWorkflowCoordinatesAndLiteralText() throws {
    let draft = ImageLayoutDraft(width: 896, height: 1152, caption: "Serum advertisement", regions: [
      .init(type: .text, caption: "Thin brown lettering", text: "GLOW SERUM",
        bounds: .init(x: 0.14, y: 0.07, width: 0.72, height: 0.1))])
    let compiled = try MOSSLayoutCompiler.compile(draft)
    let root = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(compiled.effectivePrompt.utf8)) as? [String: Any])
    let item = try XCTUnwrap((root["items"] as? [[String: Any]])?.first)
    XCTAssertEqual(item["bbox"] as? [Int], [-496, -323, -380, 323])
    XCTAssertEqual(item["text"] as? String, "GLOW SERUM")
    XCTAssertEqual(root["bbox_format"] as? String, "yxyx")
    XCTAssertNil(root["width"])
    XCTAssertEqual(compiled.width, 896)
    XCTAssertEqual(try MOSSLayoutCompiler.compile(draft).contentIdentity, compiled.contentIdentity)
  }

  func testOfficialLandscapeAndOptionalFieldsSurviveRoundtrip() throws {
    let bytes = Data(#"{"width":1536,"height":1024,"bbox_coord_space":"abs","bbox_format":"xyxy","summary":{"caption":"Campaign","meta":["poster"],"canvas":"single","palette":["cream"]},"items":[{"label":"text","caption":"Dark-green letters","text":"茶\n\"MOSS\" 🪴","tags":["headline"],"palette":["green"],"bbox":[176,224,800,864]}]}"#.utf8)
    let draft = try MOSSLayoutInterchange.importOfficialDraft(bytes)
    XCTAssertEqual(draft.summary.meta, ["poster"])
    XCTAssertEqual(draft.regions.first?.palette, ["green"])
    let rendered = try MOSSLayoutCompiler.compile(draft)
    let root = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(rendered.effectivePrompt.utf8)) as? [String: Any])
    let items = try XCTUnwrap(root["items"] as? [[String: Any]])
    XCTAssertEqual(items[0]["bbox"] as? [Int], [-288, -592, 352, 32])
    XCTAssertEqual(items[0]["text"] as? String, "茶\n\"MOSS\" 🪴")
    XCTAssertEqual(items[0]["tags"] as? [String], ["headline"])
    XCTAssertEqual((root["summary"] as? [String: Any])?["palette"] as? [String], ["cream"])
    let roundtrip = try MOSSLayoutInterchange.importOfficialDraft(MOSSLayoutInterchange.exportOfficialDraft(draft))
    XCTAssertEqual(try MOSSLayoutCompiler.compile(roundtrip).effectivePrompt, rendered.effectivePrompt)
    let compiledImport = try MOSSLayoutInterchange.importCompiled(Data(rendered.effectivePrompt.utf8), width: 1536, height: 1024)
    XCTAssertEqual(try MOSSLayoutCompiler.compile(compiledImport).effectivePrompt, rendered.effectivePrompt)
  }

  func testEmptyLayoutKeepsAuthoredPromptExactly() throws {
    let text = "  A cup marked \"茶\".\nKeep this line.  "
    XCTAssertEqual(try MOSSLayoutCompiler.compile(.init(width: 1024, height: 1024, caption: text)).effectivePrompt, text)
  }

  func testGeometryRejectsInvalidAndVanishingBoxes() throws {
    for bounds in [NormalizedLayoutRect(x: -0.01, y: 0, width: 0.2, height: 0.2),
      .init(x: 0, y: 0, width: 1.01, height: 0.1), .init(x: 0, y: 0, width: 0, height: 1),
      .init(x: 0, y: 0, width: .nan, height: 1), .init(x: 0, y: 0, width: 1e-15, height: 0.1)] {
      XCTAssertThrowsError(try ImageLayoutGeometry.pixelBounds(bounds, width: 768, height: 1344))
    }
    XCTAssertEqual(try ImageLayoutGeometry.pixelBounds(.init(x: 0, y: 0, width: 1, height: 1), width: 768, height: 1344).centeredYXYX,
      [-672, -384, 672, 384])
    XCTAssertThrowsError(try MOSSLayoutCompiler.compile(.init(width: 1023, height: 1024, caption: "test")))
  }

  func testFractionalPixelEnclosureAndBoundaryRounding() throws {
    let imported = try MOSSLayoutInterchange.importOfficialDraft(Data(#"{"width":768,"height":1344,"bbox_coord_space":"abs","bbox_format":"xyxy","summary":{"caption":"test"},"items":[{"label":"object","caption":"vase","bbox":[10.9,20.1,100.01,200.99]}]}"#.utf8))
    XCTAssertEqual(try ImageLayoutGeometry.pixelBounds(imported.regions[0].bounds, width: 768, height: 1344).centeredYXYX, [-652, -374, -471, -283])
  }

  func testInvalidFieldsDuplicateKeysAndMissingLiteralTextAreRejected() throws {
    let base = #"{"width":1024,"height":1024,"bbox_coord_space":"abs","bbox_format":"xyxy","summary":{"caption":"test"},"items":[{"label":"text","caption":"letters","bbox":[0,0,100,100]}]}"#
    XCTAssertThrowsError(try MOSSLayoutInterchange.importOfficialDraft(Data(base.utf8)))
    XCTAssertThrowsError(try MOSSLayoutInterchange.importOfficialDraft(Data(#"{"caption":"one","caption":"two"}"#.utf8)))
    XCTAssertThrowsError(try MOSSLayoutInterchange.importOfficialDraft(Data(#"{"caption":"one","children":[]}"#.utf8)))
    XCTAssertThrowsError(try MOSSLayoutInterchange.importOfficialDraft(Data(base.replacingOccurrences(of: "1024", with: "true").utf8)))
  }

  func testProportionalResizeAndViewportMapping() throws {
    let rect = NormalizedLayoutRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)
    XCTAssertEqual(try ImageLayoutGeometry.pixelBounds(rect, width: 1024, height: 2048).centeredYXYX, [-512, -256, 512, 256])
    let viewport = try LayoutViewport(canvasWidth: 896, canvasHeight: 1152, viewWidth: 1000, viewHeight: 600, zoom: 2)
    let point = viewport.viewPoint(x: 0.25, y: 0.75)
    let normalized = viewport.normalizedPoint(x: point.x, y: point.y)
    XCTAssertEqual(normalized.x, 0.25, accuracy: 1e-12)
    XCTAssertEqual(normalized.y, 0.75, accuracy: 1e-12)
  }

  func testPixaromaImportReadsSavedPropertiesAndCharacterAssignment() throws {
    let data = Data(#"{"nodes":[{"type":"PixaromaLayoutBoxes","widgets_values":[896,1152],"properties":{"layoutBoxes":{"v":1,"boxes":[{"x":0.14,"y":0.07,"w":0.72,"h":0.1,"type":"text","desc":"headline","text":"GLOW SERUM"},{"x":0.2,"y":0.3,"w":0.4,"h":0.5,"type":"obj","desc":"a person","text":""}]}}},{"type":"PixaromaPrompt","properties":{"promptState":{"text":"Saved prompt"}},"widgets_values":[""]},{"type":"MOSSImageLayoutPrompt","widgets_values":["",1024,1024,"2"]}]}"#.utf8)
    let draft = try MOSSLayoutInterchange.importPixaromaWorkflow(data)
    XCTAssertEqual(draft.width, 896)
    XCTAssertEqual(draft.height, 1152)
    XCTAssertEqual(draft.summary.caption, "Saved prompt")
    XCTAssertEqual(draft.regions.map(\.type), [.text, .character])
    XCTAssertEqual(draft.regions[0].text, "GLOW SERUM")
    let ids = draft.regions.map(\.id)
    XCTAssertEqual(try JSONDecoder().decode(ImageLayoutDraft.self, from: JSONEncoder().encode(draft)).regions.map(\.id), ids)
  }
}
