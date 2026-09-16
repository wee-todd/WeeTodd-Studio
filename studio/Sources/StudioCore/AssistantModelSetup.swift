import Foundation

/// A pinned installation offer, not a claim that arbitrary model formats can run.
public struct AssistantModelCatalog: Decodable {
  public var id: String
  public var name: String
  public var filename: String
  public var runtime: String
  public var downloadBytes: Int64
  public var requiredDiskBytes: Int64
  public var sourceURL: String
  public var licenseURL: String
  public var sha256: String
  public var notice: String
}

public struct AssistantModelInspection: Decodable {
  public var path: String
  public var status: String
  public var inferenceChecked: Bool
  public var requiresHealthCheck: Bool
  public var vision: Bool
  public var message: String
  public var selectable: Bool {
    !requiresHealthCheck && (status == "checksum_verified" || (status == "inference_checked" && inferenceChecked))
  }
}
