import AVFoundation
import Foundation

/// Keeps decoded preview audio on disk and only a few playback items resident.
@MainActor final class NativeSpeechPlayback {
  private let player = AVQueuePlayer()
  private var pending: [URL] = []
  private var enabled = false
  func start() {
    stop()
    enabled = true
  }
  private var roots = Set<URL>()
  private var observer: NSObjectProtocol?
  init() {
    observer = NotificationCenter.default.addObserver(
      forName: .AVPlayerItemDidPlayToEndTime, object: nil, queue: .main
    ) { [weak self] notification in
      Task { @MainActor in
        guard let self, let item = notification.object as? AVPlayerItem else { return }
        if let asset = item.asset as? AVURLAsset,
          self.roots.contains(asset.url.deletingLastPathComponent())
        {
          try? FileManager.default.removeItem(at: asset.url)
          self.fill()
        }
      }
    }
  }
  deinit { if let observer { NotificationCenter.default.removeObserver(observer) } }
  func enqueue(_ path: String, output: URL) {
    let root = output.deletingLastPathComponent().appendingPathComponent(
      output.lastPathComponent + "-preview"
    ).standardizedFileURL
    let url = URL(fileURLWithPath: path).standardizedFileURL
    guard enabled, url.deletingLastPathComponent() == root, url.pathExtension == "wav",
      Int(url.deletingPathExtension().lastPathComponent) != nil,
      url.resolvingSymlinksInPath().deletingLastPathComponent() == root.resolvingSymlinksInPath(),
      pending.count < 256
    else { return }
    roots.insert(root)
    pending.append(url)
    fill()
  }
  private func fill() {
    while player.items().count < 4, !pending.isEmpty {
      player.insert(AVPlayerItem(url: pending.removeFirst()), after: nil)
    }
    if !player.items().isEmpty { player.play() }
  }
  func stop() {
    enabled = false
    player.pause()
    player.removeAllItems()
    pending = []
    for root in roots { try? FileManager.default.removeItem(at: root) }
    roots = []
  }
}
