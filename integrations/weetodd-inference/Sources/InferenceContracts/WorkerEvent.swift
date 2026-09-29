import Foundation

/// Versioned JSON-lines control plane. Tensor payloads and previews travel through
/// local files/buffers, never JSON arrays or base64 strings.
public struct WorkerEvent: Codable, Sendable {
  public enum Payload: Codable, Sendable {
    case accepted
    case stageStarted(name: String)
    case progress(completed: Int, total: Int)
    case preview(relativePath: String, width: Int, height: Int)
    case stageReleased(name: String)
    case completed(artifacts: [String])
    case failed(message: String)
    case cancelled
  }
  public let version: Int
  public let jobID: UUID
  public let sequence: UInt64
  public let payload: Payload

  public init(jobID: UUID, sequence: UInt64, payload: Payload) {
    version = 1; self.jobID = jobID; self.sequence = sequence; self.payload = payload
  }

  public static func decode(_ data: Data) throws -> WorkerEvent {
    guard data.count <= 65536 else { throw ContractError.invalid("Worker event exceeds 64 KiB.") }
    let event = try JSONDecoder().decode(WorkerEvent.self, from: data)
    guard event.version == 1 else { throw ContractError.invalid("Unsupported worker protocol version.") }
    return event
  }
}

/// One validator per job. Invalid events cannot advance its sequence or state. This
/// protects the client from stale, cross-job and out-of-order worker updates.
public struct WorkerEventValidator: Sendable {
  public let jobID: UUID
  public private(set) var isTerminal = false
  public private(set) var residentStage: String?
  private var nextSequence: UInt64 = 0
  private var accepted = false
  private var completedUnits = 0
  private var totalUnits: Int?

  public init(jobID: UUID) { self.jobID = jobID }

  public mutating func accept(_ event: WorkerEvent) throws {
    guard event.version == 1, event.jobID == jobID, event.sequence == nextSequence,
          nextSequence < UInt64.max, !isTerminal else {
      throw ContractError.invalid("Worker event has the wrong job, version, sequence or terminal state.")
    }
    switch event.payload {
    case .accepted:
      guard !accepted else { throw ContractError.invalid("Worker accepted the job twice.") }
      accepted = true
    default:
      guard accepted else { throw ContractError.invalid("Worker event arrived before job acceptance.") }
      try advance(event.payload)
    }
    nextSequence += 1
  }

  private mutating func advance(_ payload: WorkerEvent.Payload) throws {
    switch payload {
    case .accepted: break
    case .stageStarted(let name):
      guard residentStage == nil, !name.isEmpty, name.utf8.count <= 128 else {
        throw ContractError.invalid("Release the previous weighted stage before starting another.")
      }
      residentStage = name; completedUnits = 0; totalUnits = nil
    case .progress(let completed, let total):
      guard residentStage != nil, total > 0, completed >= completedUnits, completed <= total,
            totalUnits == nil || totalUnits == total else {
        throw ContractError.invalid("Worker stage progress is invalid or regressed.")
      }
      completedUnits = completed; totalUnits = total
    case .preview(let path, let width, let height):
      guard residentStage != nil, Self.validArtifactPath(path), width > 0, height > 0,
            width <= 2048, height <= 2048 else {
        throw ContractError.invalid("Worker preview must use a relative artifact path and bounded dimensions.")
      }
    case .stageReleased(let name):
      guard residentStage == name else { throw ContractError.invalid("Worker released a stage that is not resident.") }
      residentStage = nil; completedUnits = 0; totalUnits = nil
    case .completed(let artifacts):
      guard residentStage == nil, !artifacts.isEmpty, artifacts.count <= 128,
            artifacts.allSatisfy(Self.validArtifactPath) else {
        throw ContractError.invalid("Worker completion requires released stages and valid output artifacts.")
      }
      isTerminal = true
    case .failed(let message):
      guard !message.isEmpty, message.utf8.count <= 8192 else { throw ContractError.invalid("Worker failure message is invalid.") }
      isTerminal = true // residency remains until release is proven by worker exit
    case .cancelled:
      isTerminal = true
    }
  }

  private static func validArtifactPath(_ path: String) -> Bool {
    guard !path.isEmpty, path.utf8.count <= 4096, !path.hasPrefix("/"),
          !path.contains(":"), !path.contains("\\"), !path.unicodeScalars.contains(where: { $0.value < 32 }) else { return false }
    return path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy {
      !$0.isEmpty && $0 != "." && $0 != ".."
    }
  }
}
