import AppKit
import SwiftUI

/// Give the reference editor room to work, constrained to the current display.
struct ReferenceEditorWindow: NSViewRepresentable {
  func makeNSView(context: Context) -> SizingView { SizingView() }
  func updateNSView(_ view: SizingView, context: Context) {}

  final class SizingView: NSView {
    private weak var sizedWindow: NSWindow?
    override func viewDidMoveToWindow() {
      super.viewDidMoveToWindow()
      guard let window, window !== sizedWindow else { return }
      sizedWindow = window
      DispatchQueue.main.async { [weak self, weak window] in
        guard let self, let window, self.window === window else { return }
        let available = (window.screen ?? NSScreen.main)?.visibleFrame.size ?? NSSize(width: 1440, height: 1000)
        window.styleMask.insert(.resizable)
        window.contentMinSize = NSSize(width: 980, height: 680)
        window.setContentSize(NSSize(width: min(1480, available.width - 80), height: min(960, available.height - 100)))
      }
    }
  }
}
