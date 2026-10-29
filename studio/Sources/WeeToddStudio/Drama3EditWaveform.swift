import AVFoundation
import StudioCore
import SwiftUI

struct Drama3EditWaveform: View {
  @EnvironmentObject var store: StudioStore
  let source: String
  @State private var peaks: [Float] = []
  @State private var duration: Double = 0
  @State private var problem: String?
  var start: Double { store.project.voiceDraft?.dramaEditStart ?? 0 }
  var end: Double { store.project.voiceDraft?.dramaEditEnd ?? min(1, duration) }
  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      GeometryReader { geometry in
        Canvas { context, size in
          guard duration > 0, !peaks.isEmpty else { return }
          let left = max(0, min(1, start / duration)) * size.width
          let right = max(0, min(1, end / duration)) * size.width
          context.fill(
            Path(CGRect(x: left, y: 0, width: max(2, right - left), height: size.height)),
            with: .color(.accentColor.opacity(0.2)))
          var waveform = Path()
          for (i, peak) in peaks.enumerated() {
            let x = Double(i) * size.width / Double(peaks.count)
            let height = Double(peak) * size.height * 0.45
            waveform.move(to: CGPoint(x: x, y: size.height / 2 - height))
            waveform.addLine(to: CGPoint(x: x, y: size.height / 2 + height))
          }
          context.stroke(waveform, with: .color(.secondary), lineWidth: 1)
        }.background(Color.secondary.opacity(0.06)).contentShape(Rectangle())
          .gesture(
            DragGesture(minimumDistance: 0).onEnded { drag in
              guard duration > 0, !store.operationBusy else { return }
              let a = max(0, min(duration, drag.startLocation.x / geometry.size.width * duration))
              let b = max(0, min(duration, drag.location.x / geometry.size.width * duration))
              let insertion = store.project.voiceDraft?.dramaEditOperation == "insert"
              store.change {
                $0.voiceDraft?.dramaEditStart = min(a, b)
                $0.voiceDraft?.dramaEditEnd = insertion ? min(a, b) : max(a, b)
              }
            })
      }.frame(height: 72).accessibilityLabel(
        "Speech waveform. Drag to select an edit range; exact seconds can be entered above.")
      HStack {
        Text("\(duration,specifier:"%.2f") s · drag to replace a range, click for an insertion")
          .font(.caption).foregroundStyle(.secondary)
        Button("Audition range") {
          let item = AVPlayerItem(url: URL(fileURLWithPath: source))
          item.forwardPlaybackEndTime = CMTime(
            seconds: max(start + 0.1, end), preferredTimescale: 44100)
          store.musicPlayer.replaceCurrentItem(with: item)
          store.musicPlayer.seek(to: CMTime(seconds: start, preferredTimescale: 44100))
          store.musicPlayer.play()
        }
      }
      if let problem { Text(problem).font(.caption).foregroundStyle(.orange) }
    }.task(id: source) {
      do {
        let result = try await Task.detached { try Self.read(source) }.value
        guard !Task.isCancelled else { return }
        peaks = result.0
        duration = result.1
        problem = nil
      } catch { problem = error.localizedDescription }
    }
  }
  nonisolated private static func read(_ path: String) throws -> ([Float], Double) {
    let file = try AVAudioFile(forReading: URL(fileURLWithPath: path))
    let format = file.processingFormat
    guard file.length > 0, file.length <= Int64(format.sampleRate * 600) else {
      throw StudioError.invalid("Choose speech up to ten minutes long.")
    }
    var peaks = [Float](repeating: 0, count: 512)
    let length = file.length
    while file.framePosition < length {
      try Task.checkCancellation()
      let begin = file.framePosition
      let count = AVAudioFrameCount(min(65536, length - begin))
      let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: count)!
      try file.read(into: buffer, frameCount: count)
      guard let channels = buffer.floatChannelData, buffer.frameLength > 0 else { break }
      for i in 0..<Int(buffer.frameLength) {
        let slot = min(511, Int((begin + Int64(i)) * 512 / length))
        for channel in 0..<Int(format.channelCount) {
          let value = abs(channels[channel][i])
          if value.isFinite { peaks[slot] = max(peaks[slot], min(1, value)) }
        }
      }
    }
    return (peaks, Double(length) / format.sampleRate)
  }
}
