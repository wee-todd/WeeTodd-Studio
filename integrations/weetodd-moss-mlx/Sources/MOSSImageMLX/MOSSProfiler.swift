import Foundation
import MLX

public struct MOSSOperatorTiming: Codable, Sendable {
  public var count = 0
  public var seconds = 0.0
  public var bytes = 0
}

/// Opt-in synchronized operator groups. Nested groups overlap; diagnostic
/// synchronization changes timing overhead and is excluded from ordinary runs.
final class MOSSProfiler {
  private(set) var records: [String: MOSSOperatorTiming] = [:]
  func record(_ name: String, seconds: Double, bytes: Int = 0) {
    var value = records[name] ?? .init(); value.count += 1; value.seconds += seconds; value.bytes += bytes
    records[name] = value
  }
  func measure<T>(_ name: String, _ body: () throws -> T) rethrows -> T {
    Stream.defaultStream(.gpu).synchronize()
    let start = Date(); let result = try body()
    Stream.defaultStream(.gpu).synchronize()
    record(name, seconds: Date().timeIntervalSince(start)); return result
  }
  func evaluate(_ name: String, _ value: MLXArray) -> MLXArray {
    measure(name) { eval(value); return value }
  }
}
