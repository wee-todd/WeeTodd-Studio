import Foundation

/// Explicit source-movie workflow; ordinary manual generation defaults stay unchanged.
public struct LTX25MovieUpscaleSettings:Codable,Equatable,Sendable {
  public enum Mode:String,Codable,CaseIterable,Identifiable,Sendable {
    case latentOnly="latent_only",refine,pixelSpatial="pixel_spatial"
    public var id:String { rawValue }
    public var label:String { self == .latentOnly ? "Learned 2× only" : self == .refine ? "Learned 2× + refine" : "Pixel-Spatial guided 2× + refine" }
  }
  public enum SizePolicy:String,Codable,CaseIterable,Identifiable,Sendable {
    case fitNearest="fit_nearest_32",centerCrop="center_crop_32",strict="strict_32"
    public var id:String { rawValue }
  }
  public enum Anchors:String,Codable,CaseIterable,Identifiable,Sendable {
    case none,first,firstLast="first_last"
    public var id:String { rawValue }
  }
  public enum AudioPolicy:String,Codable,CaseIterable,Identifiable,Sendable {
    case source,sidecar,silence
    public var id:String { rawValue }
  }
  public var mode:Mode = .refine
  public var sizePolicy:SizePolicy = .fitNearest
  public var refinementStrength:Double = 0.35
  public var anchors:Anchors = .first
  public var anchorStrength:Double = 0.7
  public var pixelStrength:Double = 1
  public var pixelSpatialAdapterPath:String?
  public var audioPolicy:AudioPolicy = .source
  public var maximumAudioDriftSeconds:Double = 0.05
  public var chunking:Bool = false
  public var chunkFrameMegapixelBudget:Double = 260
  public var resume:Bool = false
  public var keepChunks:Bool = false
  public var experimentalEnabled:Bool = false
  public init() {}
  public func validate() throws {
    guard experimentalEnabled else { throw StudioError.invalid("Enable experimental native movie upscaling explicitly. Its real-model quality is not yet qualified.") }
    guard refinementStrength.isFinite,(0.05...0.909375).contains(refinementStrength),
      anchorStrength.isFinite,(0...1).contains(anchorStrength),pixelStrength.isFinite,(0.05...2).contains(pixelStrength),
      maximumAudioDriftSeconds.isFinite,(0...0.5).contains(maximumAudioDriftSeconds),
      chunkFrameMegapixelBudget.isFinite,chunkFrameMegapixelBudget>0,!resume || chunking else {
      throw StudioError.invalid("Movie upscaling has invalid strength, audio drift or explicit chunk/resume settings.")
    }
    guard mode != .latentOnly || anchors == .none else {
      throw StudioError.invalid("Learned 2× only does not use endpoint conditioning. Select no anchors explicitly.")
    }
    if let path=pixelSpatialAdapterPath {
      guard mode == .pixelSpatial,path.hasPrefix("/"),path.utf8.count<=4096,!path.utf8.contains(0) else {
        throw StudioError.invalid("An explicit Pixel-Spatial adapter requires Pixel-Spatial mode and an absolute local path.")
      }
    }
  }
}
