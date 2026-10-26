import AppKit
import Foundation
import StudioCore
import SwiftUI
import XCTest
@testable import WeeToddStudio

final class LoRAPromptMetadataEditorTests: XCTestCase {
  @MainActor func testTypingInHostedEditorDoesNotSaveOrMutateTheProject() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let store = StudioStore(dataDirectory: root, restoreSession: false)
    store.addClip(); let before = store.project
    let key = LoRAPromptKey.native("/tmp/weapon.safetensors")
    let editor = LoRAPromptMetadataEditor(name: "Weapon Combat", initial: .init(
      triggers: ["BUNNY"], notes: "Old notes", sourceURL: "https://example.com/lora")) {
      store.saveLoRAPromptMetadata($0, key: key)
    }
    _ = NSApplication.shared
    let host = NSHostingView(rootView: editor.environment(\.colorScheme, .dark))
    let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 610, height: 540), styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false; window.appearance = NSAppearance(named: .darkAqua); window.contentView = host
    defer { window.close() }
    host.frame = window.contentView!.bounds; host.layoutSubtreeIfNeeded()
    RunLoop.main.run(until: Date().addingTimeInterval(0.1)); host.layoutSubtreeIfNeeded()
    func descendants(_ view: NSView) -> [NSView] { [view] + view.subviews.flatMap(descendants) }
    let views = descendants(host)
    let trigger = try XCTUnwrap(views.compactMap { $0 as? NSTextView }.first { $0.string == "BUNNY" })
    let notes = try XCTUnwrap(views.compactMap { $0 as? NSTextView }.first { $0.string == "Old notes" })
    trigger.insertText("BUNNY\nr3tr0sks", replacementRange: NSRange(location: 0, length: trigger.string.utf16.count))
    notes.insertText("Choose a clear action sequence.", replacementRange: NSRange(location: 0, length: notes.string.utf16.count))
    RunLoop.main.run(until: Date().addingTimeInterval(0.1))
    if let target = ProcessInfo.processInfo.environment["WEETODD_LORA_PROMPT_SNAPSHOT"] {
      let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
      host.cacheDisplay(in: host.bounds, to: bitmap)
      try XCTUnwrap(bitmap.representation(using: .png, properties: [:])).write(to: URL(fileURLWithPath: target))
    }
    XCTAssertEqual(trigger.string, "BUNNY\nr3tr0sks")
    XCTAssertEqual(notes.string, "Choose a clear action sequence.")
    XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("lora-prompt-metadata.json").path))
    XCTAssertEqual(store.project, before)
    XCTAssertFalse(window.isVisible)
  }
}
