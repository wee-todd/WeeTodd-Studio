import Foundation
import InferenceContracts

public enum ReCaError: LocalizedError {
  case invalid(String)
  public var errorDescription: String? { if case .invalid(let message) = self { return message }; return nil }
}

public struct ReCaGeometry: Sendable {
  public let sourceWidth: Int, sourceHeight: Int
  public let width: Int, height: Int, paddedWidth: Int, paddedHeight: Int
  public var tokenWidth: Int { paddedWidth / 32 }
  public var tokenHeight: Int { paddedHeight / 32 }
  public var latentWidth: Int { paddedWidth / 16 }
  public var latentHeight: Int { paddedHeight / 16 }
  public init(sourceWidth: Int, sourceHeight: Int, scale: Double) throws {
    guard (1...16384).contains(sourceWidth), (1...16384).contains(sourceHeight),
      [1.5, 2, 4].contains(scale) else { throw ReCaError.invalid("Invalid ReCaVSR source size or scale.") }
    self.sourceWidth = sourceWidth; self.sourceHeight = sourceHeight
    width = Int(floor(Double(sourceWidth) * scale + 0.5))
    height = Int(floor(Double(sourceHeight) * scale + 0.5))
    paddedWidth = (width + 31) / 32 * 32; paddedHeight = (height + 31) / 32 * 32
  }
  func sourceIndex(_ index: Int, axisSize: Int) -> Int {
    let padding = (axisSize + 31) / 32 * 32 - axisSize
    return index < axisSize ? index : padding < axisSize ? 2 * axisSize - 2 - index : axisSize - 1
  }
}
