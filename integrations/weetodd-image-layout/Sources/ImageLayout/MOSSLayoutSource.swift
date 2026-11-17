import Foundation

/// Explicit generation input, independent of the editor guide and prompt compiler.
public struct MOSSLayoutSource: Codable, Equatable, Sendable {
  public var path, sha256: String
  public var enabled: Bool
  /// Actual starting flow sigma; the requested number of active steps is retained.
  public var noiseLevel: Double
  public init(path: String, sha256: String, enabled: Bool = true, noiseLevel: Double = 0.08) {
    self.path = path; self.sha256 = sha256; self.enabled = enabled; self.noiseLevel = noiseLevel
  }
  public func validate() throws {
    guard path.hasPrefix("/"), !path.utf8.contains(0), sha256.count == 64,
      sha256.allSatisfy({ $0.isHexDigit && !$0.isUppercase }),
      noiseLevel.isFinite, noiseLevel > 0, noiseLevel < 1 else {
      throw ImageLayoutError.invalid("Source reconstruction needs a local image, its SHA-256 and change strength strictly between zero and one.")
    }
  }
}
