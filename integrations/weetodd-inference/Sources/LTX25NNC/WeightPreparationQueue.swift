import Foundation

struct PreparedWeights {
  let index: Int
  let tensors: [String: [Float]]
  let bytes: UInt64
  let seconds: Double
}

/// At most one admitted CPU block. No graph, tensor or Metal API is used here.
/// The owning inference thread serializes submit/take/drain. A job's immutable
/// provider is borrowed only until drain; its result is synchronized by the group.
final class WeightPreparationQueue {
  private final class Job: @unchecked Sendable {
    let index: Int
    let layout: [(String, [Int])]
    let provider: (Int, String, [Int]) throws -> [Float]
    let group = DispatchGroup()
    private let lock = NSLock()
    private var cancelled = false
    var result: Result<PreparedWeights, Error>?
    init(index: Int, layout: [(String, [Int])], provider: @escaping (Int, String, [Int]) throws -> [Float]) {
      self.index = index; self.layout = layout; self.provider = provider
      group.enter()
    }
    func cancel() { lock.lock(); cancelled = true; lock.unlock() }
    func checkCancellation() throws {
      lock.lock(); let stopped = cancelled; lock.unlock()
      if stopped { throw CancellationError() }
    }
    func run() {
      defer { group.leave() }
      result = Result {
        let began = Date()
        var tensors: [String: [Float]] = [:], bytes: UInt64 = 0
        for (name, shape) in layout {
          try checkCancellation()
          let values = try provider(index, name, shape)
          guard values.count == shape.reduce(1, *) else { throw BlockError.invalid("Prepared weight shape mismatch.") }
          tensors[name] = values; bytes += UInt64(values.count) * 4
        }
        try checkCancellation()
        return PreparedWeights(index: index, tensors: tensors, bytes: bytes, seconds: Date().timeIntervalSince(began))
      }
    }
  }
  private let queue = DispatchQueue(label: "wee-todd.ltx.weight-prepare", qos: .userInitiated)
  private let layout: [(String, [Int])]
  let admittedBytes: UInt64
  private var job: Job?
  var pending: Bool { job != nil }

  init(layout: [(String, [Int])], maximumBytes: UInt64) throws {
    guard !layout.isEmpty, Set(layout.map(\.0)).count == layout.count,
      maximumBytes <= 2 * 1024 * 1024 * 1024 else { throw BlockError.invalid("Invalid preparation budget/layout.") }
    var total: UInt64 = 0
    for (_, shape) in layout {
      guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 && $0 <= 131072 }) else {
        throw BlockError.invalid("Invalid prepared tensor shape.")
      }
      var size: UInt64 = 4
      for d in shape {
        let product = size.multipliedReportingOverflow(by: UInt64(d))
        guard !product.overflow else { throw BlockError.invalid("Prepared tensor size overflow.") }
        size = product.partialValue
      }
      guard total <= maximumBytes, size <= maximumBytes - total else {
        throw BlockError.invalid("CPU preparation exceeds its admitted byte budget.")
      }
      total += size
    }
    self.layout = layout; admittedBytes = total
  }
  func submit(index: Int, provider: @escaping (Int, String, [Int]) throws -> [Float]) throws {
    guard job == nil else { throw BlockError.invalid("Preparation queue is occupied.") }
    let next = Job(index: index, layout: layout, provider: provider)
    job = next; queue.async { next.run() }
  }
  func take() throws -> PreparedWeights {
    guard let current = job else { throw BlockError.invalid("No pending preparation.") }
    current.group.wait(); job = nil
    guard let result = current.result else { throw BlockError.invalid("Missing preparation result.") }
    try Task.checkCancellation()
    return try result.get()
  }
  func cancelPending() { job?.cancel() }

  func drain() {
    guard let current = job else { return }
    current.cancel(); current.group.wait(); job = nil
  }
  deinit { drain() }
}
