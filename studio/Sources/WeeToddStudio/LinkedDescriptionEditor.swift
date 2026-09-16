import AppKit
import StudioCore
import SwiftUI

/// Native editable text with display-only object links and per-range macOS tooltips.
struct LinkedDescriptionEditor: NSViewRepresentable {
  @Binding var text: String
  var targets: [DescriptionLinkTarget]
  var mentions: [DescriptionMention] = []
  var sourceDescription: String? = nil
  var editable: Bool
  var accessibilityLabel: String
  var onNavigate: (String) -> Void
  @Environment(\.isEnabled) private var enabled

  func makeCoordinator() -> Coordinator { Coordinator(self) }
  func makeNSView(context: Context) -> NSScrollView {
    let scroll = NSScrollView()
    scroll.borderType = .bezelBorder
    scroll.hasVerticalScroller = true; scroll.autohidesScrollers = true
    let view = NSTextView()
    view.isRichText = false; view.importsGraphics = false; view.allowsUndo = true
    view.isAutomaticLinkDetectionEnabled = false
    view.isVerticallyResizable = true; view.isHorizontallyResizable = false
    view.autoresizingMask = [.width]
    view.textContainer?.widthTracksTextView = true
    view.textContainerInset = NSSize(width: 5, height: 6)
    view.font = .systemFont(ofSize: NSFont.systemFontSize)
    view.backgroundColor = .textBackgroundColor
    view.linkTextAttributes = [.foregroundColor: NSColor.linkColor, .underlineStyle: NSUnderlineStyle.single.rawValue]
    view.delegate = context.coordinator
    view.setAccessibilityLabel(accessibilityLabel)
    scroll.documentView = view
    return scroll
  }
  func updateNSView(_ scroll: NSScrollView, context: Context) {
    context.coordinator.parent = self
    guard let view = scroll.documentView as? NSTextView else { return }
    view.isEditable = editable && enabled; view.isSelectable = true
    if view.string != text {
      let selection = view.selectedRange()
      context.coordinator.updating = true
      view.string = text
      view.setSelectedRange(NSRange(location: min(selection.location, (text as NSString).length), length: 0))
      context.coordinator.updating = false
    }
    context.coordinator.decorate(view)
  }
  final class Coordinator: NSObject, NSTextViewDelegate {
    var parent: LinkedDescriptionEditor
    var updating = false
    var destinations: [String: String] = [:]
    init(_ parent: LinkedDescriptionEditor) { self.parent = parent }
    func textDidChange(_ notification: Notification) {
      guard !updating, let view = notification.object as? NSTextView else { return }
      parent.text = view.string
      decorate(view)
    }
    func decorate(_ view: NSTextView) {
      guard !view.hasMarkedText(), let storage = view.textStorage else { return }
      updating = true; defer { updating = false }
      let all = NSRange(location: 0, length: storage.length)
      storage.beginEditing()
      for key in [NSAttributedString.Key.link, .toolTip, .underlineStyle] { storage.removeAttribute(key, range: all) }
      storage.addAttributes([.font: NSFont.systemFont(ofSize: NSFont.systemFontSize), .foregroundColor: NSColor.textColor], range: all)
      destinations.removeAll()
      for (index, link) in DescriptionLinks.ranges(in: view.string, targets: parent.targets, mentions: parent.mentions, sourceDescription: parent.sourceDescription).enumerated() {
        let address = "weetodd-object:\(index)"
        destinations[address] = link.targetID
        storage.addAttributes([.link: address, .toolTip: link.tooltip,
          .foregroundColor: NSColor.linkColor, .underlineStyle: NSUnderlineStyle.single.rawValue], range: link.range)
      }
      storage.endEditing()
      view.typingAttributes = [.font: NSFont.systemFont(ofSize: NSFont.systemFontSize), .foregroundColor: NSColor.textColor]
    }
    func textView(_ textView: NSTextView, clickedOnLink link: Any, at charIndex: Int) -> Bool {
      let address = (link as? URL)?.absoluteString ?? (link as? String) ?? ""
      if let id = destinations[address] { parent.onNavigate(id) }
      // Links are internal only; never let AppKit dispatch a URL to another application.
      return true
    }
  }
}
