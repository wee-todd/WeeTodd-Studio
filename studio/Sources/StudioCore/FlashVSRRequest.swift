import CryptoKit
import Foundation

public struct FlashVSRRuntime: Codable, Equatable, Sendable {
  public var vpipePath = "", modelRoot = "", supervisorPath = "", ffmpegPath = "", ffprobePath = ""
  public var leasePath: String = ""
  public var swiftWorkerPath: String?
  public init() {}
  private enum CodingKeys: String, CodingKey, CaseIterable {
    case vpipePath, modelRoot, supervisorPath, ffmpegPath, ffprobePath, leasePath, swiftWorkerPath
  }
  public init(from decoder: Decoder) throws {
    try FlashVSRStrictObject.check(decoder, allowed: Set(CodingKeys.allCases.map(\.rawValue)))
    let fields = try decoder.container(keyedBy: CodingKeys.self)
    vpipePath = try fields.decodeIfPresent(String.self, forKey: .vpipePath) ?? ""
    modelRoot = try fields.decodeIfPresent(String.self, forKey: .modelRoot) ?? ""
    supervisorPath = try fields.decodeIfPresent(String.self, forKey: .supervisorPath) ?? ""
    ffmpegPath = try fields.decodeIfPresent(String.self, forKey: .ffmpegPath) ?? ""
    ffprobePath = try fields.decodeIfPresent(String.self, forKey: .ffprobePath) ?? ""
    leasePath = try fields.decodeIfPresent(String.self, forKey: .leasePath) ?? ""
    swiftWorkerPath = try fields.decodeIfPresent(String.self, forKey: .swiftWorkerPath)
  }
}
public struct VideoUpscaleRequest: Codable, Equatable, Sendable {
  public var format = "weetodd-video-upscale-v1"
  public var sourcePath: String
  public var sourceSHA256: String?
  public var firstFrame = 0
  public var frameCount: Int?
  public var settings: FlashVSRSettings
  public var runtime: FlashVSRRuntime
  private enum CodingKeys: String, CodingKey, CaseIterable { case format, sourcePath, sourceSHA256, firstFrame, frameCount, settings, runtime }
  public init(from decoder: Swift.Decoder) throws {
    try FlashVSRStrictObject.check(decoder, allowed: Set(CodingKeys.allCases.map(\.rawValue)))
    let values = try decoder.container(keyedBy: CodingKeys.self)
    format = try values.decode(String.self, forKey: .format)
    sourcePath = try values.decode(String.self, forKey: .sourcePath)
    sourceSHA256 = try values.decodeIfPresent(String.self, forKey: .sourceSHA256)
    firstFrame = try values.decodeIfPresent(Int.self, forKey: .firstFrame) ?? 0
    frameCount = try values.decodeIfPresent(Int.self, forKey: .frameCount)
    settings = try values.decode(FlashVSRSettings.self, forKey: .settings)
    runtime = try values.decode(FlashVSRRuntime.self, forKey: .runtime)
  }
  public init(sourcePath: String, settings: FlashVSRSettings = .init(), runtime: FlashVSRRuntime) {
    self.sourcePath = sourcePath; self.settings = settings; self.runtime = runtime
  }
}
public struct FlashVSRResolvedPlan: Codable, Sendable {
  public let request: VideoUpscaleRequest
  public let source: FlashVSRFile
  public let media: FlashVSRMediaInfo
  public let components: FlashVSRComponents
  public let engineFiles: [FlashVSRFile]
  public let frames: Int
  public let temporaryBytes: Int64
  public var fingerprint: String { get throws {
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
    return SHA256.hash(data: try encoder.encode(self)).map { String(format: "%02x", $0) }.joined()
  } }
}
public struct FlashVSRReceipt: Codable, Sendable {
  public var format = "weetodd-video-upscale-result-v1"
  public let requestFingerprint: String
  public let plan: FlashVSRResolvedPlan
  public let video: FlashVSRFile
  public let restored: FlashVSRFile
  public let audio: FlashVSRFile?
  public var sourcePreview: FlashVSRFile?
  public let elapsedSeconds: Double
}
public struct FlashVSREvent: Codable, Sendable {
  public let stage: String
  public let message: String
  public let fraction: Double?
  public init(stage: String, message: String, fraction: Double? = nil) {
    self.stage = stage; self.message = message; self.fraction = fraction
  }
}
public protocol FlashVSRBackend {
  func preflight(request: VideoUpscaleRequest) async throws -> FlashVSRResolvedPlan
  func run(plan: FlashVSRResolvedPlan, directory: URL, emit: @escaping @Sendable (FlashVSREvent) -> Void) async throws -> FlashVSRReceipt
}
/// Native sequence stages publish lossless pixels; delivery encoding belongs
/// to the sequence after every selected model has completed.
public protocol FlashVSRIntermediateBackend: FlashVSRBackend {
  func runIntermediate(plan: FlashVSRResolvedPlan, directory: URL,
    emit: @escaping @Sendable (FlashVSREvent) -> Void) async throws -> FlashVSRReceipt
}
