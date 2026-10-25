import AppKit
import Foundation
import StudioCore
import SwiftUI
import XCTest
@testable import WeeToddStudio

final class GenerationInspectorSnapshotTests: XCTestCase {
  @MainActor func testIngredientsBasicHostsWithoutChangingTheSavedRecipe() throws {
    let fixture = try ingredientsFixture()
    defer { try? FileManager.default.removeItem(at: fixture.directory) }
    try capture(ClipInspector(clip: fixture.clip, detail: .basic), fixture: fixture,
                name: "ingredients-basic", height: 1500)
  }

  @MainActor func testH3FirstLastBasicHostsWithoutChangingItsFramesOrLoRA() throws {
    let fixture = try h3Fixture()
    defer { try? FileManager.default.removeItem(at: fixture.directory) }
    try capture(ClipInspector(clip: fixture.clip, detail: .basic), fixture: fixture,
                name: "h3-first-last-basic", height: 1600)
  }

  @MainActor func testH3TwoStageStrengthIsEditableInBasicGenerationSettings() throws {
    var fixture = try h3Fixture()
    defer { try? FileManager.default.removeItem(at: fixture.directory) }
    fixture.clip.generationSelection?.h3Joint = .init(twoStage:H3TwoStageSettings())
    fixture.store.project.clips = [fixture.clip]
    try capture(ClipInspector(clip:fixture.clip,detail:.basic),fixture:fixture,
      name:"h3-two-stage-strength-basic",height:1900,
      requiredFields:["Second-pass strength":0.35])
  }

  @MainActor func testIngredientsAllHostsWithoutReplacingItsExplicitSettings() throws {
    let fixture = try ingredientsFixture()
    defer { try? FileManager.default.removeItem(at: fixture.directory) }
    try capture(ClipInspector(clip: fixture.clip, detail: .all), fixture: fixture,
                name: "ingredients-all", height: 3000)
  }

  @MainActor func testDrawThingsAllHostsItsConnectionModelAndSavedConfiguration() async throws {
    var fixture = try h3Fixture()
    defer { try? FileManager.default.removeItem(at: fixture.directory) }
    fixture.clip.engine = .drawThings
    fixture.clip.name = "Draw Things · first and last"
    fixture.clip.profileID = "auto"
    fixture.clip.generationSelection = GenerationSelection(task: "fflf", preset: .custom)
    fixture.clip.attachments.removeAll { $0.role == .lora }
    fixture.clip.drawThings = DrawThingsSelection(profileID: "fixture-server", modelID: "h3-fl2va",
      modelFamily: "minimaxh3", configuration: ["steps": .integer(50), "guidanceScale": .number(1),
        "shift": .number(12), "audioShift": .number(3), "fps": .integer(24), "sampler": .integer(13)],
      loras: [DrawThingsLoRA(modelID: "h3-style", weight: 0.65, enabled: true)])
    fixture.clip.drawThings?.modelModifier = "fl2va"
    fixture.store.project.clips = [fixture.clip]
    fixture.store.project.assets.removeAll { $0.kind == .lora }
    fixture.store.profiles = []; fixture.store.generationDescriptions = [:]
    let connection = DrawThingsConnection(id: "fixture-server", name: "Local Draw Things · fixture")
    fixture.store.drawThingsConnections = [connection]
    await fixture.store.drawThingsDiscovery.load(connection) {
      ["models": [["id": "h3-fl2va", "name": "MiniMax H3 FL2VA"]],
       "capabilities": ["h3-fl2va": ["family": "minimaxh3", "modifier": "fl2va",
         "operations": ["video": [:]]]],
       "loras": [["id": "h3-style", "name": "H3 cinematic style", "family": "minimaxh3",
         "compatibleModelIDs": ["h3-fl2va"]]]]
    }
    try capture(ClipInspector(clip: fixture.clip, detail: .all), fixture: fixture,
                name: "draw-things-all", height: 2000)
  }

  private struct Fixture {
    var directory: URL
    var store: StudioStore
    var clip: Clip
  }

  @MainActor private func capture<Content: View>(_ content: Content, fixture: Fixture,
                                                name: String, height: CGFloat, requiredFields:[String:Double] = [:]) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    let before = try encoder.encode(fixture.clip)
    _ = NSApplication.shared
    let width: CGFloat = 420
    let root = content
      .environmentObject(fixture.store)
      .environment(\.colorScheme, .dark)
      .frame(width: width - 32, alignment: .leading)
      .padding(16)
      .frame(width: width, height: height, alignment: .topLeading)
      .background(Theme.panel)
    let host = NSHostingView(rootView: root)
    let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: width, height: height),
                          styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.appearance = NSAppearance(named: .darkAqua)
    window.contentView = host
    defer { window.close() }
    host.frame = NSRect(x: 0, y: 0, width: width, height: height)
    host.layoutSubtreeIfNeeded()
    RunLoop.main.run(until: Date().addingTimeInterval(0.08))
    host.layoutSubtreeIfNeeded()
    host.displayIfNeeded()
    var fields:[String:NSTextField]=[:]
    func inspect(_ view:NSView) {
      if let field=view as? NSTextField,let label=field.placeholderString,
        requiredFields[label] != nil {fields[label]=field}
      for child in view.subviews {inspect(child)}
    }
    if !requiredFields.isEmpty {inspect(host)}
    XCTAssertFalse(window.isVisible, "Snapshot hosting must never show a desktop window")
    let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
    host.cacheDisplay(in: host.bounds, to: bitmap)
    let png = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
    XCTAssertGreaterThanOrEqual(bitmap.pixelsWide, Int(width))
    XCTAssertGreaterThanOrEqual(bitmap.pixelsHigh, Int(height))
    XCTAssertGreaterThan(png.count, 1024)
    let decoded = try XCTUnwrap(NSBitmapImageRep(data: png))
    XCTAssertEqual(decoded.pixelsWide, bitmap.pixelsWide)
    XCTAssertEqual(decoded.pixelsHigh, bitmap.pixelsHigh)

    let after = try XCTUnwrap(fixture.store.selectedClip)
    XCTAssertEqual(try encoder.encode(after), before,
                   "Hosting generation controls must preserve the exact saved clip")
    XCTAssertEqual(after.profileID, fixture.clip.profileID)
    XCTAssertEqual(after.generationSelection?.preset, fixture.clip.generationSelection?.preset)
    XCTAssertEqual(after.inferredTask, fixture.clip.inferredTask)
    XCTAssertEqual(after.seed, fixture.clip.seed)
    XCTAssertEqual(after.attachments, fixture.clip.attachments)
    XCTAssertFalse(fixture.store.canUndo)

    if let destination = ProcessInfo.processInfo.environment["WEETODD_INSPECTOR_SNAPSHOT_DIR"] {
      guard destination.hasPrefix("/") else {
        throw StudioError.invalid("WEETODD_INSPECTOR_SNAPSHOT_DIR must be an absolute directory.")
      }
      let directory = URL(fileURLWithPath: destination, isDirectory: true)
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
      try png.write(to: directory.appendingPathComponent(name + ".png"), options: .atomic)
    }
    for (label,value) in requiredFields {
      let field=try XCTUnwrap(fields[label],"Missing editable field in basic inspector: \(label)")
      XCTAssertTrue(field.isEditable);XCTAssertTrue(field.isEnabled)
      XCTAssertEqual(field.doubleValue,value,accuracy:0.000001)
    }
  }

  @MainActor private func ingredientsFixture() throws -> Fixture {
    let directory = try temporaryDirectory()
    var clip = Clip(name: "Diner conversation", engine: .ltx25)
    clip.profileID = directory.appendingPathComponent("ingredients-ancestral.json").path
    clip.prompt = "Two friends sit across a diner table. The man on the left lifts his coffee mug; the man on the right smiles and replies. Keep both described characters distinct."
    clip.soundscape = "Quiet diner chatter, dishes and a coffee mug touching the table."
    clip.seed = 20261009
    var sheet = MediaAsset(name: "Two character reference sheet", kind: .image,
                           path: try imageFixture(directory, name: "reference-sheet", color: .systemTeal).path,
                           scope: .clip, owner: clip.id)
    sheet.width = 256; sheet.height = 160; sheet.thumbnail = sheet.path
    var attachment = Attachment(assetID: sheet.id, role: .control)
    attachment.controlType = "ingredients_reference_sheet"
    attachment.description = "The left character has dark hair and a blue shirt. The right character has a beard and a brown jacket."
    clip.attachments = [attachment]
    clip.generationSelection = GenerationSelection(task: "control", preset: .custom)
    let generation: [String: Any] = [
      "supportedTasks": ["control"], "referenceFamily": "ingredients",
      "pipelineMode": "distilled", "singleStageAvailable": true,
      "singleStageEnabled": true, "singleStageMethods": ["euler_ancestral", "euler_ancestral_cfg_pp"],
      "ordinaryKeyframesAvailable": false, "dfrEnabled": false,
      "controls": controls(evaluations: 8, refinement: 0, editableSteps: false),
      "presets": [["id": "custom", "name": "Custom", "description": "Preserve imported recipe settings."]],
    ]
    let recipe: [String: Any] = ["engine": "ltx25", "name": "LTX 2.5 Ingredients · Euler ancestral",
      "conditioning": ["task": "control"],
      "config": ["ic_lora_single_stage": true, "single_stage_sampler": "euler_ancestral", "stage1_steps": 8, "stage2_steps": 0],
      "components": ["ic_loras": [["/fixture/ingredients.safetensors", 1.2]]]]
    try JSONSerialization.data(withJSONObject: recipe).write(to: URL(fileURLWithPath: clip.profileID))
    let store = try makeStore(directory: directory, clip: clip, assets: [sheet], generation: generation,
                              profileName: "LTX 2.5 Ingredients · Euler ancestral")
    return Fixture(directory: directory, store: store, clip: clip)
  }

  @MainActor private func h3Fixture() throws -> Fixture {
    let directory = try temporaryDirectory()
    var clip = Clip(name: "Cup lift · first and last", engine: .h3)
    clip.profileID = directory.appendingPathComponent("h3-fl2va.json").path
    clip.prompt = "A man sits at a wooden table and raises the white coffee cup from the first image to the last image. The camera stays steady, with warm afternoon window light."
    clip.seed = 20261008
    clip.generationSelection = GenerationSelection(task: "fflf", preset: .custom)
    clip.generationSelection?.steps = 4
    clip.generationSelection?.h3SamplingMethod = .euler
    var first = MediaAsset(name: "Cup on the table", kind: .image,
                           path: try imageFixture(directory, name: "first", color: .systemOrange).path,
                           scope: .clip, owner: clip.id)
    var last = MediaAsset(name: "Cup raised", kind: .image,
                          path: try imageFixture(directory, name: "last", color: .systemYellow).path,
                          scope: .clip, owner: clip.id)
    first.width = 256; first.height = 160; first.thumbnail = first.path
    last.width = 256; last.height = 160; last.thumbnail = last.path
    var adapter = MediaAsset(name: "LightX2V H3 Turbo", kind: .lora,
                             path: directory.appendingPathComponent("turbo.safetensors").path,
                             scope: .clip, owner: clip.id)
    adapter.loraModel = .h3; adapter.loraProfile = "turbo"
    var lora = Attachment(assetID: adapter.id, role: .lora)
    lora.strength = 1; lora.enabled = true
    clip.attachments = [Attachment(assetID: first.id, role: .first),
                        Attachment(assetID: last.id, role: .last, time: clip.duration - 1 / 24), lora]
    let generation: [String: Any] = [
      "supportedTasks": ["i2v", "fflf"], "samplingMethod": "euler",
      "videoDecodePrecision": "float32", "videoDecodePrecisionEditable": true,
      "attentionPolicy": "dense", "attentionPolicyEditable": true,
      "transformerWeightCacheGB": 0, "transformerWeightCacheEditable": true,
      "controls": controls(evaluations: 4, refinement: nil, editableSteps: true, cfg: nil),
      "presets": [["id": "custom", "name": "Custom", "description": "Use the admitted profile settings."]],
    ]
    let store = try makeStore(directory: directory, clip: clip, assets: [first, last, adapter],
                              generation: generation, profileName: "MiniMax H3 FL2VA · Q8")
    return Fixture(directory: directory, store: store, clip: clip)
  }

  @MainActor private func makeStore(directory: URL, clip: Clip, assets: [MediaAsset],
                                    generation: [String: Any], profileName: String) throws -> StudioStore {
    let store = StudioStore(dataDirectory: directory, restoreSession: false,
      invocation: { command, _, _, _ in
        XCTFail("Inspector fixture attempted execution: \(command)")
        throw StudioError.invalid("Inspector snapshots never execute models or servers.")
      })
    store.runtime = RuntimeSettings(root: "/fixture/unavailable", pythonPath: "/fixture/unavailable/python",
                                    profilesDirectory: directory.path)
    store.project.clips = [clip]; store.project.assets = assets; store.selectedClipID = clip.id
    let descriptor = try JSONDecoder().decode(GenerationDescriptor.self,
      from: JSONSerialization.data(withJSONObject: generation))
    store.profiles = [ModelProfile(id: clip.profileID, name: profileName, engine: clip.engine.rawValue,
                                  task: clip.inferredTask, generation: descriptor)]
    store.generationDescriptions[clip.id] = ["studioEngine": clip.engine.rawValue,
      "studioTask": clip.inferredTask, "studioProfile": clip.profileID,
      "studioInput": store.generationRequestKey(for: clip), "generation": generation, "warnings": []]
    return store
  }

  private func controls(evaluations: Int, refinement: Int?, editableSteps: Bool, cfg: Double? = 1) -> [String: Any] {
    var result: [String: Any] = ["evaluations": evaluations,
      "stepsEditable": editableSteps, "refinementStepsEditable": false,
      "cfgEditable": false, "shiftEditable": false,
      "stepsExplanation": "Preserve the admitted recipe's evaluation count.",
      "cfgExplanation": "Guidance is built into the admitted model.",
      "shiftExplanation": "This native adapter does not expose a Shift override."]
    if let refinement { result["refinementSteps"] = refinement }
    if let cfg { result["cfg"] = cfg }
    return result
  }

  private func temporaryDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
  }

  @MainActor private func imageFixture(_ directory: URL, name: String, color: NSColor) throws -> URL {
    let bitmap = try XCTUnwrap(NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 256, pixelsHigh: 160,
      bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
      colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0))
    let context = try XCTUnwrap(NSGraphicsContext(bitmapImageRep: bitmap))
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = context
    color.setFill(); NSRect(x: 0, y: 0, width: 256, height: 160).fill()
    NSColor.white.withAlphaComponent(0.8).setFill()
    NSBezierPath(roundedRect: NSRect(x: 56, y: 32, width: 64, height: 92), xRadius: 16, yRadius: 16).fill()
    NSColor.black.withAlphaComponent(0.35).setFill()
    NSBezierPath(roundedRect: NSRect(x: 144, y: 32, width: 64, height: 92), xRadius: 16, yRadius: 16).fill()
    NSGraphicsContext.restoreGraphicsState()
    let destination = directory.appendingPathComponent(name + ".png")
    try XCTUnwrap(bitmap.representation(using: .png, properties: [:])).write(to: destination)
    return destination
  }
}
