import Drama3Contracts
import Foundation
import MLX

public enum Drama3AudioEdit {
  public static func partition(_ codes: MLXArray, start: Double, end: Double) throws -> (
    prefix: MLXArray, suffix: MLXArray
  ) {
    try Drama3AudioCodec.validate(codes)
    guard start.isFinite, end.isFinite, start >= 0, end >= start,
      end <= Double(codes.dim(1)) * 2048 / 44100 + 1e-7
    else { throw Drama3Error.invalid("Choose an edit range inside the source take.") }
    // A zero-width insertion uses one shared codec boundary; independent rounding
    // would accidentally delete the frame underneath an ordinary waveform click.
    let insertion = start == end
    let a = min(
      codes.dim(1),
      Int(insertion ? (start * 44100 / 2048).rounded() : floor(start * 44100 / 2048 + 1e-8)))
    let b = insertion ? a : min(codes.dim(1), Int(ceil(end * 44100 / 2048 - 1e-8)))
    return (codes[0..., ..<a], codes[0..., b...])
  }
}
