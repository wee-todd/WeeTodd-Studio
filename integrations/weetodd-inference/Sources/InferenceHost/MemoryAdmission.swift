import Foundation

/// The peak simultaneous allocation estimate for one weighted stage. Prefetch is
/// accounted separately so overlapping upload cannot silently double the weight budget.
public struct StageMemory: Sendable, Equatable {
  public let weights: UInt64
  public let activations: UInt64
  public let scratch: UInt64
  public let prefetch: UInt64
  public let totalBytes: UInt64

  public init(weights: UInt64, activations: UInt64, scratch: UInt64, prefetch: UInt64) throws {
    var total: UInt64 = 0
    for component in [weights, activations, scratch, prefetch] {
      let sum = total.addingReportingOverflow(component)
      guard !sum.overflow else { throw HostError.invalidMemoryEstimate }
      total = sum.partialValue
    }
    self.weights = weights; self.activations = activations
    self.scratch = scratch; self.prefetch = prefetch; totalBytes = total
  }
}

/// Owned by the parent coordinator and shared across workers. Reservations are
/// admission estimates, not measurements or an OS allocation limit. Keep a lease
/// until GPU work is synchronized and the stage/worker has actually released memory.
public actor MemoryAdmission {
  public struct Lease: Sendable {
    fileprivate let id: UUID
    public let jobID: UUID
    public let stage: String
    public let bytes: UInt64
  }

  public let usableBytes: UInt64
  public private(set) var reservedBytes: UInt64 = 0
  private var active: [UUID: Lease] = [:]

  public init(capacityBytes: UInt64, reserveBytes: UInt64) throws {
    guard capacityBytes > reserveBytes else { throw HostError.invalidMemoryEstimate }
    usableBytes = capacityBytes - reserveBytes
  }

  public func acquire(jobID: UUID, stage: String, estimate: StageMemory) throws -> Lease {
    try Task.checkCancellation()
    let available = usableBytes - reservedBytes
    guard estimate.totalBytes <= available else {
      throw HostError.insufficientMemory(required: estimate.totalBytes, available: available)
    }
    let lease = Lease(id: UUID(), jobID: jobID, stage: stage, bytes: estimate.totalBytes)
    active[lease.id] = lease
    reservedBytes += lease.bytes
    return lease
  }

  public func release(_ lease: Lease) {
    guard let owned = active.removeValue(forKey: lease.id) else { return }
    reservedBytes -= owned.bytes
  }

  public func withReservation<T: Sendable>(jobID: UUID, stage: String, estimate: StageMemory,
    operation: @Sendable () async throws -> T) async throws -> T {
    let lease = try acquire(jobID: jobID, stage: stage, estimate: estimate)
    defer { release(lease) }
    let result = try await operation()
    try Task.checkCancellation()
    return result
  }
}
