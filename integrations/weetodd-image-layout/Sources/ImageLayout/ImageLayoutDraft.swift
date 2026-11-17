import Foundation

public enum ImageLayoutError: Error, LocalizedError, Equatable {
  case invalid(String)
  public var errorDescription: String? { if case .invalid(let message) = self { message } else { nil } }
}

public enum ImageLayoutRegionType: String, Codable, CaseIterable, Sendable {
  case text, object, character
}

public struct NormalizedLayoutRect: Codable, Equatable, Sendable {
  public var x, y, width, height: Double
  public init(x: Double, y: Double, width: Double, height: Double) {
    self.x = x; self.y = y; self.width = width; self.height = height
  }
}

public struct ImageLayoutSummary: Codable, Equatable, Sendable {
  public var caption: String
  public var meta: [String]?
  public var canvas: String?
  public var palette: [String]?
  public init(caption: String, meta: [String]? = nil, canvas: String? = nil, palette: [String]? = nil) {
    self.caption = caption; self.meta = meta; self.canvas = canvas; self.palette = palette
  }
}

public struct ImageLayoutRegion: Codable, Equatable, Identifiable, Sendable {
  public enum AnalysisRole: String, Codable, Sendable { case face, canvasBackground }
  public var id: UUID
  /// Editor ownership only. MOSS receives the child boxes, not this metadata.
  public var parentID: UUID?
  /// Analysis/editor structure. Never emitted as a model-facing layout field.
  public var analysisRole: AnalysisRole?
  public var type: ImageLayoutRegionType
  public var caption: String
  public var text: String?
  public var bounds: NormalizedLayoutRect
  public var tags: [String]?
  public var palette: [String]?
  public init(id: UUID = UUID(), type: ImageLayoutRegionType, caption: String, text: String? = nil,
    bounds: NormalizedLayoutRect, tags: [String]? = nil, palette: [String]? = nil) {
    self.id = id; self.type = type; self.caption = caption; self.text = text
    self.bounds = bounds; self.tags = tags; self.palette = palette
  }
}

public struct ImageLayoutDraft: Codable, Equatable, Sendable {
  public var width, height: Int
  public var summary: ImageLayoutSummary
  public var regions: [ImageLayoutRegion]
  /// An editor guide only; never interpreted as an inference image input.
  public var guideImagePath: String?
  public var guideOpacity: Double?
  public var analysis: ImageLayoutAnalysisProvenance?
  public var sourceReconstruction: MOSSLayoutSource?
  public init(width: Int, height: Int, caption: String, regions: [ImageLayoutRegion] = []) {
    self.width = width; self.height = height; summary = .init(caption: caption); self.regions = regions
  }
}
