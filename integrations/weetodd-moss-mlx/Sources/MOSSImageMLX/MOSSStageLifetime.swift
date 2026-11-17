import Foundation

final class MOSSStageLifetime {
  enum Stage: String { case encoding, conditioning, transformer, decoding }
  private(set) var active: Stage?
  private(set) var released: [String] = []
  func withStage<T>(_ stage: Stage, _ body: () throws -> T) throws -> T {
    guard active == nil else { throw MOSSError("Weighted MOSS stages cannot overlap.") }
    active = stage
    defer { active = nil; released.append(stage.rawValue) }
    return try body()
  }
}
