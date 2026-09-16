import AppKit
import StudioCore
import SwiftUI
import XCTest
@testable import WeeToddStudio

final class LinkedDescriptionEditorTests: XCTestCase {
  @MainActor func testNativeTooltipRefreshNavigationAndPlainTextPreservation() {
    let text = "Mara wears the copper jacket."
    var destination: String?
    var target = DescriptionLinkTarget(id: "jacket", name: "Copper jacket", description: "High collar and black zipper.")
    let editor = LinkedDescriptionEditor(text: .constant(text), targets: [target], editable: true,
      accessibilityLabel: "Description", onNavigate: { destination = $0 })
    let coordinator = editor.makeCoordinator()
    let view = NSTextView(); view.string = text
    view.setSelectedRange(NSRange(location: 5, length: 0))
    coordinator.decorate(view)
    let index = (text as NSString).range(of: "copper jacket").location
    XCTAssertEqual(view.textStorage?.attribute(.toolTip, at: index, effectiveRange: nil) as? String,
                   "Copper jacket\n\nHigh collar and black zipper.")
    let link = view.textStorage!.attribute(.link, at: index, effectiveRange: nil)!
    XCTAssertTrue(coordinator.textView(view, clickedOnLink: link, at: index))
    XCTAssertEqual(destination, "jacket")
    XCTAssertEqual(view.string, text)
    XCTAssertEqual(view.selectedRange(), NSRange(location: 5, length: 0))
    target.description = "Updated reference description."
    coordinator.parent.targets = [target]; coordinator.decorate(view)
    XCTAssertEqual(view.textStorage?.attribute(.toolTip, at: index, effectiveRange: nil) as? String,
                   "Copper jacket\n\nUpdated reference description.")
    coordinator.parent.targets = []; coordinator.decorate(view)
    XCTAssertNil(view.textStorage?.attribute(.link, at: index, effectiveRange: nil))
    XCTAssertNil(view.textStorage?.attribute(.toolTip, at: index, effectiveRange: nil))
    XCTAssertEqual(view.string, text)
  }
}
