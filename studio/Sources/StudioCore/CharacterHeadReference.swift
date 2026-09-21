import Foundation

public struct CharacterHeadSelection: Codable, Hashable, Sendable {
  public var crop: PanelPixelRect
  public var foregroundInstanceIndex: Int?
  public init(crop: PanelPixelRect, foregroundInstanceIndex: Int? = nil) {
    self.crop = crop; self.foregroundInstanceIndex = foregroundInstanceIndex
  }
}

public struct CharacterHeadReference: Codable, Equatable, Sendable {
  public var originalAssetID: UUID
  public var headCropAssetID: UUID
  public var maskAssetID: UUID
  public var rgbaCutoutAssetID: UUID
  public var whiteMatteAssetID: UUID
  public var sourcePath: String
  public var headCropPath: String
  public var maskPath: String
  public var rgbaCutoutPath: String
  public var whiteMattePath: String
  public var crop: PanelPixelRect
  public var sourceSHA256: String
  public var preprocessingVersion: String
  public var matteRGB: [UInt8]
  public var artifactSHA256: [String: String]

  public init(originalAssetID: UUID, headCropAssetID: UUID, maskAssetID: UUID,
    rgbaCutoutAssetID: UUID, whiteMatteAssetID: UUID, sourcePath: String = "",
    headCropPath: String = "", maskPath: String = "", rgbaCutoutPath: String = "",
    whiteMattePath: String = "", crop: PanelPixelRect = .init(x: 0, y: 0, width: 1, height: 1),
    sourceSHA256: String, preprocessingVersion: String, matteRGB: [UInt8] = [255, 255, 255],
    artifactSHA256: [String: String] = ["headCrop": "", "mask": "", "rgbaCutout": "", "whiteMatte": ""]) {
    self.originalAssetID = originalAssetID; self.headCropAssetID = headCropAssetID
    self.maskAssetID = maskAssetID; self.rgbaCutoutAssetID = rgbaCutoutAssetID
    self.whiteMatteAssetID = whiteMatteAssetID; self.sourcePath = sourcePath
    self.headCropPath = headCropPath; self.maskPath = maskPath
    self.rgbaCutoutPath = rgbaCutoutPath; self.whiteMattePath = whiteMattePath
    self.crop = crop; self.sourceSHA256 = sourceSHA256
    self.preprocessingVersion = preprocessingVersion; self.matteRGB = matteRGB
    self.artifactSHA256 = artifactSHA256
  }
}
