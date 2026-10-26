import Foundation

public enum H3ImageDecoderMode: String, Codable, CaseIterable, Identifiable {
  case qualityFiveFrames = "quality_five_frames"
  case singleLatent = "single_latent"
  public var id: String { rawValue }
  public var label: String { self == .qualityFiveFrames ? "Quality · original VAE, five frames" : "Experimental · single-image VAE" }
}

public enum H3ImageCandidate: String, Codable, CaseIterable, Identifiable {
  case first, middle, last
  public var id: String { rawValue }
}

public struct H3ImageSettings: Codable, Equatable {
  public var decoderMode: H3ImageDecoderMode = .qualityFiveFrames
  public var frameSelection: H3ImageCandidate = .middle
  public var imageVAEPath = ""
  public var loras: [LoRAMember] = []
  public init() {}
}
