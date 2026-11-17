import Foundation

public enum MOSSSchedule {
  public static func simple(steps: Int, shift: Float = 5) throws -> [Float] {
    guard (1...1000).contains(steps), shift.isFinite, shift > 0, shift <= 100 else { throw MOSSError("Invalid simple schedule parameters.") }
    return (0..<steps).map { i in
      let t = Float(1000 - Int(Double(i) * 1000 / Double(steps))) / 1000
      return shift * t / (1 + (shift - 1) * t)
    } + [0]
  }
  /// The pinned release's Float32 linspace and nearest training-grid schedule; terminal sigma is zero.
  public static func sigmas(steps: Int, shift: Float = 5) throws -> [Float] {
    guard (1...1000).contains(steps), shift.isFinite, shift > 0, shift <= 100 else { throw MOSSError("Use a finite positive shift and 1–1000 requested steps.") }
    func grid(_ count: Int, _ shift: Float) -> [Float] {
      let end = Float(Double(shift) / (1000 + Double(shift) - 1))
      let step = count > 1 ? (end - 1) / Float(count - 1) : 0
      return (0..<count).map { i in
        let v: Float = i < count / 2 || count == 1 ? Float(1).addingProduct(step, Float(i))
          : end.addingProduct(-step, Float(count - i - 1))
        return shift * v / (1 + (shift - 1) * v)
      }
    }
    let training = Array(grid(1000, 1).reversed())
    let snapped = grid(steps, shift).map { sigma -> Float in
      var lower = 0, upper = training.count
      while lower < upper {
        let mid = (lower + upper) / 2
        if training[mid] < sigma { lower = mid + 1 } else { upper = mid }
      }
      let high = min(lower, 999), low = max(0, high - 1)
      return abs(sigma - training[low]) <= abs(training[high] - sigma) ? training[low] : training[high]
    }
    guard zip(snapped, snapped.dropFirst()).allSatisfy({ $0 > $1 }), snapped.last! > 0 else {
      throw MOSSError("Requested steps collapse on the MOSS training grid; choose fewer steps.")
    }
    return snapped + [0]
  }
}
