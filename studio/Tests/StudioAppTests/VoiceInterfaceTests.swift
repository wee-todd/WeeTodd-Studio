import AppKit
import StudioCore
import SwiftUI
import Vision
import XCTest
@testable import WeeToddStudio

final class VoiceInterfaceTests: XCTestCase {
  @MainActor func testVoiceWorkspaceShowsTheSelectedNativeDramaFamily() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.openVoice()
    let text = try renderedText(VoiceEditor().environmentObject(store),
                               name: "voice-family", width: 1150, height: 1600)
    XCTAssertTrue(text.contains("drama3"), text)
    XCTAssertTrue(text.contains("native"), text)
    XCTAssertTrue(store.showVoice)
    XCTAssertNil(store.error)
  }

  @MainActor func testSpeechSettingsShowDramaAndBothModelSetupControls() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    var settings = VoiceModelSettings()
    settings.register(.init(engine: .fishDrama3, name: "Balanced", path: "/missing/native/model"))
    store.runtime.voiceModels = settings
    let text = try renderedText(VoiceModelSettingsView().environmentObject(store),
                               name: "speech-model-settings", width: 650, height: 900)
    XCTAssertTrue(text.contains("fishdrama3"), text)
    XCTAssertTrue(text.contains("addinstalledmodel"), text)
    XCTAssertTrue(text.contains("downloadspeechmodel"), text)
    XCTAssertTrue(text.contains("completescenecontrols"), text)
  }

  /// Render real controls without a visible desktop window. SwiftUI's virtual accessibility
  /// children are unavailable for hidden windows, so recognize the rendered labels instead.
  @MainActor private func renderedText<Content: View>(_ content: Content, name: String,
                                                     width: CGFloat, height: CGFloat) throws -> String {
    _ = NSApplication.shared
    let host = NSHostingView(rootView: content.environment(\.colorScheme, .dark)
      .frame(width: width, height: height, alignment: .topLeading).background(Theme.panel))
    let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: width, height: height),
                          styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.appearance = NSAppearance(named: .darkAqua)
    window.contentView = host
    defer { window.close() }
    host.frame = NSRect(x: 0, y: 0, width: width, height: height)
    host.layoutSubtreeIfNeeded()
    RunLoop.main.run(until: Date().addingTimeInterval(0.1))
    host.layoutSubtreeIfNeeded()
    host.displayIfNeeded()
    XCTAssertFalse(window.isVisible)
    let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
    host.cacheDisplay(in: host.bounds, to: bitmap)
    if let output = ProcessInfo.processInfo.environment["WEETODD_VOICE_SNAPSHOT_DIR"] {
      let directory = URL(fileURLWithPath: output, isDirectory: true)
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
      try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
        .write(to: directory.appendingPathComponent(name + ".png"))
    }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.recognitionLanguages = ["en-US"]
    request.usesLanguageCorrection = false
    try VNImageRequestHandler(cgImage: XCTUnwrap(bitmap.cgImage)).perform([request])
    return (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }
      .joined(separator: "\n").lowercased().filter { !$0.isWhitespace }
  }
}
