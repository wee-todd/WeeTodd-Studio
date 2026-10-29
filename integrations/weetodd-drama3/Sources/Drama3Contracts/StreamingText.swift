import Foundation

public struct Drama3TextChunk: Sendable, Equatable {
  public let text: String
  public let lookahead: String?
}
/// Bounded producer input. A chunk becomes runnable when its lookahead arrives or input closes.
public actor Drama3TextInput {
  private var queue: [String] = []
  private var closed = false, cancelled = false, totalBytes = 0
  private let capacity: Int
  private var waiter: CheckedContinuation<Void, Never>?
  public init(capacity: Int = 64) { self.capacity = max(2, min(capacity, 64)) }
  public var ready: Bool { queue.count >= 2 || closed || cancelled }
  public func append(_ text: String) throws {
    guard !closed, !cancelled else { throw Drama3Error.invalid("Text input has already finished.") }
    let chunks = try Drama3Text.chunks(text)
    guard queue.count + chunks.count <= capacity, totalBytes + text.utf8.count <= 1_000_000 else {
      throw Drama3Error.invalid(
        "Text queue is full. Wait for a segment before appending more text.")
    }
    queue += chunks
    totalBytes += text.utf8.count
    wake()
  }
  public func finish() {
    closed = true
    wake()
  }
  public func cancel() {
    cancelled = true
    queue = []
    wake()
  }
  private func wake() {
    if ready {
      waiter?.resume()
      waiter = nil
    }
  }
  public func next() async throws -> Drama3TextChunk? {
    try await withTaskCancellationHandler(
      operation: {
        while !ready { await withCheckedContinuation { waiter = $0 } }
        try Task.checkCancellation()
        if cancelled { throw CancellationError() }
        guard !queue.isEmpty else { return nil }
        let text = queue.removeFirst()
        return .init(text: text, lookahead: queue.first)
      }, onCancel: { Task { await self.cancel() } })
  }
}
