import Foundation

public enum FlashVSRResourcePlan {
  public static func temporaryBytes(media: FlashVSRMediaInfo, settings: FlashVSRSettings, frames: Int) throws -> Int64 {
    let canvas = try settings.canvas()
    guard frames > 0, frames <= 100_000 else { throw StudioError.invalid("The selected upscale interval exceeds its frame budget.") }
    let prepared = Int64(media.width) * Int64(media.height) * 8
    let restoredAndDelivery = Int64(canvas.width) * Int64(canvas.height) * 6
    var bytes = Int64(frames) * (prepared + restoredAndDelivery) + 512 * 1024 * 1024
    if settings.preserveAudio && media.hasAudio {
      guard let rate = media.audioRate, let channels = media.audioChannels,
        (8000...192000).contains(rate), (1...8).contains(channels) else {
        throw StudioError.invalid("The source audio format is unsupported. Normalize it or turn off Keep audio before inference.")
      }
      bytes += Int64(ceil(Double(frames) / media.fps.value * Double(rate * channels * 4 * 2)))
    }
    return bytes
  }
}
