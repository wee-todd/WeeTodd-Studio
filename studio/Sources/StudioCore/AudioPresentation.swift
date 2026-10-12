import Foundation

public struct AudioCaption: Codable, Equatable, Identifiable {
  public var id = UUID()
  public var start: Double
  public var end: Double
  public var text: String
  public init(start: Double, end: Double, text: String) { self.start = start; self.end = end; self.text = text }
}
public struct AudioPresentation: Codable, Equatable {
  public var enabled = true
  public var style = "waveform"
  public var source = "mix"
  public var color = "63DEC2"
  public var background = "101827"
  public var size: Double = 0.4
  public var position: Double = 0.5
  public var sensitivity: Double = 1
  public var smoothing: Double = 0.7
  public var voiceCaptions = false
  public var lyrics = false
  public var showBothCaptions = false
  public var captionColor = "FFFFFF"
  public var captionSize: Double = 48
  public var captionPosition = "bottom"
  public init() {}
  public func validate() throws {
    guard ["waveform", "spectrum"].contains(style), ["mix", "voice", "music", "voiceAndMusic"].contains(source),
      [color, background, captionColor].allSatisfy({ $0.range(of: "^[0-9a-fA-F]{6}$", options: .regularExpression) != nil }),
      size.isFinite, (0.1...0.8).contains(size), position.isFinite, (0.2...0.8).contains(position),
      sensitivity.isFinite, (0.1...4).contains(sensitivity), smoothing.isFinite, (0...1).contains(smoothing),
      captionSize.isFinite, (16...96).contains(captionSize), ["bottom", "center", "top"].contains(captionPosition) else {
      throw StudioError.invalid("Check visualizer colors, size, sensitivity and caption style.")
    }
  }
}
public struct AudioVisualFeatures: Codable, Equatable {
  public var hop: Double
  public var waveform: [[Double]]
  public var spectrum: [[Double]]
  public init(hop: Double, waveform: [[Double]], spectrum: [[Double]]) {
    self.hop = hop; self.waveform = waveform; self.spectrum = spectrum
  }
  public func values(at time: Double, style: String, smoothing: Double) -> [Double] {
    let rows = style == "spectrum" ? spectrum : waveform
    guard time.isFinite, time >= 0, hop > 0, hop.isFinite, !rows.isEmpty else { return [] }
    let point = min(Double(rows.count - 1), time / hop)
    let i = Int(point), next = min(i + 1, rows.count - 1)
    let blend = (point - Double(i)) * smoothing
    return zip(rows[i], rows[next]).map { $0 * (1 - blend) + $1 * blend }
  }
}
extension StudioProject {
  public func hasVideo(at time: Double) -> Bool {
    guard let span = TimelinePlaybackPlan(project: self).span(at: time) else { return false }
    return !span.path.isEmpty
  }
  public func captionText(at time: Double) -> String {
    guard let settings = audioPresentation else { return "" }
    var voice: [String] = [], music: [String] = []
    let solo = audioTracks.contains(where: \.solo)
    for region in resolvedAudio where time >= region.start && time < region.start + region.duration {
      guard let track = audioTracks.first(where: { $0.id == region.trackID }), !track.muted,
        !solo || track.solo else { continue }
      guard region.instrumental != true else { continue }
      let local = time - region.start + region.sourceIn
      let text = (region.captions ?? []).filter { local >= $0.start && local < $0.end }.map(\.text).joined(separator: " ")
      guard !text.isEmpty else { continue }
      if track.role == .voice && settings.voiceCaptions { voice.append(text) }
      if track.role == .music && settings.lyrics { music.append(text) }
    }
    if !voice.isEmpty && !settings.showBothCaptions { return voice.joined(separator: "\n") }
    return (voice + music).joined(separator: "\n")
  }
}
