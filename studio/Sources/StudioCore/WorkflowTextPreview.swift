/// Bound text passed to layout without changing saved outputs or explicit copies.
public struct WorkflowTextPreview {
  public let fullText: String
  public let text: String
  public let isTruncated: Bool

  public init(_ fullText: String) {
    self.fullText = fullText
    // Count scalars, not grapheme clusters: one cluster can contain arbitrarily
    // many combining marks. Layout must remain bounded even for malformed output.
    let scalars = fullText.unicodeScalars
    let end = scalars.index(scalars.startIndex, offsetBy: 8_000, limitedBy: scalars.endIndex) ?? scalars.endIndex
    text = String(scalars[..<end])
    isTruncated = end != scalars.endIndex
  }
}
