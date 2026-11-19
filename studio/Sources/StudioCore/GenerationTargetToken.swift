import Foundation
import CryptoKit

public struct GenerationTargetToken: Equatable, Hashable {
  public var documentSessionID: UUID
  public var workspaceKey: String
  public var semanticRevision: String
  public var leaseID: UUID?
  public var memberIDs: [UUID]
  public init(documentSessionID: UUID, workspaceKey: String, semanticRevision: String, leaseID: UUID? = nil, memberIDs: [UUID] = []) {
    self.documentSessionID = documentSessionID; self.workspaceKey = workspaceKey; self.semanticRevision = semanticRevision
    self.leaseID = leaseID; self.memberIDs = memberIDs
  }
  public static func revision<T: Encodable>(_ value: T) throws -> String {
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
    return SHA256.hash(data: try encoder.encode(value)).map { String(format: "%02x", $0) }.joined()
  }
}
