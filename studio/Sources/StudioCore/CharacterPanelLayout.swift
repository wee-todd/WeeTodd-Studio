import Foundation

public struct PanelPixelRect: Codable, Hashable, Sendable {
  public var x: Int
  public var y: Int
  public var width: Int
  public var height: Int

  public init(x: Int, y: Int, width: Int, height: Int) {
    self.x = x; self.y = y; self.width = width; self.height = height
  }

  public var maxX: Int { let result = x.addingReportingOverflow(width); return result.overflow ? Int.max : result.partialValue }
  public var maxY: Int { let result = y.addingReportingOverflow(height); return result.overflow ? Int.max : result.partialValue }
  public func isValid(inWidth imageWidth: Int, height imageHeight: Int) -> Bool {
    x >= 0 && y >= 0 && width > 0 && height > 0 && x <= imageWidth && y <= imageHeight
      && width <= imageWidth - x && height <= imageHeight - y
  }
  public func contains(_ other: Self) -> Bool {
    x <= other.x && y <= other.y && maxX >= other.maxX && maxY >= other.maxY
  }
  public func intersects(_ other: Self) -> Bool {
    x < other.maxX && other.x < maxX && y < other.maxY && other.y < maxY
  }
}

public enum CharacterPanelRole: String, Codable, CaseIterable, Sendable { case front, side, back, closeUp }
public enum CharacterPanelDetectionStatus: String, Codable, Sendable { case detected, needsReview }

public struct CharacterPanelEvidence: Codable, Hashable, Sendable {
  public var foregroundBounds: [PanelPixelRect]
  public var leftGutterScore: Double?
  public var rightGutterScore: Double?
  public var rectangleSupport: Double
  public init(foregroundBounds: [PanelPixelRect], leftGutterScore: Double? = nil,
              rightGutterScore: Double? = nil, rectangleSupport: Double = 0) {
    self.foregroundBounds = foregroundBounds; self.leftGutterScore = leftGutterScore
    self.rightGutterScore = rightGutterScore; self.rectangleSupport = rectangleSupport
  }
}

public struct DetectedCharacterPanel: Identifiable, Codable, Hashable, Sendable {
  public var id: UUID
  public var role: CharacterPanelRole
  public var sourcePixelRect: PanelPixelRect
  public var evidence: CharacterPanelEvidence
  public var detectionRevision: Int
  public init(id: UUID = UUID(), role: CharacterPanelRole, sourcePixelRect: PanelPixelRect,
              evidence: CharacterPanelEvidence, detectionRevision: Int) {
    self.id = id; self.role = role; self.sourcePixelRect = sourcePixelRect
    self.evidence = evidence; self.detectionRevision = detectionRevision
  }
}

public struct CharacterPanelDetection: Codable, Equatable, Sendable {
  public var sourceSHA256: String
  public var sourceOrientation: Int
  public var detectorVersion: String
  public var candidates: [DetectedCharacterPanel]
  public var status: CharacterPanelDetectionStatus
  public var diagnostics: [String]
  public init(sourceSHA256: String, sourceOrientation: Int, detectorVersion: String,
              candidates: [DetectedCharacterPanel], status: CharacterPanelDetectionStatus,
              diagnostics: [String]) {
    self.sourceSHA256 = sourceSHA256; self.sourceOrientation = sourceOrientation
    self.detectorVersion = detectorVersion; self.candidates = candidates
    self.status = status; self.diagnostics = diagnostics
  }
}

public enum CharacterPanelLayout {
  public static let detectorVersion = "vision-panels-v1"

  public static func detect(imageWidth: Int, imageHeight: Int,
    foregroundComponents: [PanelPixelRect], edgeColumns: [Double],
    rectangleHints: [PanelPixelRect] = [], detectionRevision: Int,
    sourceSHA256: String = "", sourceOrientation: Int = 1) -> CharacterPanelDetection {
    let components = foregroundComponents.filter { $0.isValid(inWidth: imageWidth, height: imageHeight) }.sorted { $0.x < $1.x }
    guard components.count == 4 else {
      return CharacterPanelDetection(sourceSHA256: sourceSHA256, sourceOrientation: sourceOrientation,
        detectorVersion: detectorVersion, candidates: [], status: .needsReview,
        diagnostics: ["Panel detection requires exactly four distinct horizontal foreground groups; found \(components.count)."])
    }
    var boundaries = [0]
    var scores: [Double] = []
    var missingGutterEvidence = false
    for index in 0..<3 {
      let low = components[index].maxX
      let high = components[index + 1].x
      guard high > low else {
        return CharacterPanelDetection(sourceSHA256: sourceSHA256, sourceOrientation: sourceOrientation,
          detectorVersion: detectorVersion, candidates: [], status: .needsReview,
          diagnostics: ["Foreground groups touch or overlap near panel \(index + 1); review crop boundaries."])
      }
      let range = low..<high
      let samples = range.map { $0 < edgeColumns.count ? edgeColumns[$0] : 1.0 }
      let minimum = samples.min() ?? 1
      let quiet = range.filter { ($0 < edgeColumns.count ? edgeColumns[$0] : 1.0) <= 0.12 }
      missingGutterEvidence = missingGutterEvidence || quiet.isEmpty
      let boundary = quiet.isEmpty ? (low + high) / 2 : quiet[(quiet.count - 1) / 2]
      boundaries.append(boundary)
      scores.append(max(0, 1 - minimum))
    }
    boundaries.append(imageWidth)
    let roles = CharacterPanelRole.allCases
    let candidates = (0..<4).map { index in
      let rect = PanelPixelRect(x: boundaries[index], y: 0,
        width: boundaries[index + 1] - boundaries[index], height: imageHeight)
      let rectangleSupport = rectangleHints.contains { $0.contains(components[index]) } ? 1.0 : 0.0
      return DetectedCharacterPanel(role: roles[index], sourcePixelRect: rect,
        evidence: CharacterPanelEvidence(foregroundBounds: [components[index]],
          leftGutterScore: index > 0 ? scores[index - 1] : nil,
          rightGutterScore: index < 3 ? scores[index] : nil, rectangleSupport: rectangleSupport),
        detectionRevision: detectionRevision)
    }
    guard zip(candidates, components).allSatisfy({ $0.sourcePixelRect.contains($1) }) else {
      return CharacterPanelDetection(sourceSHA256: sourceSHA256, sourceOrientation: sourceOrientation,
        detectorVersion: detectorVersion, candidates: [], status: .needsReview,
        diagnostics: ["A proposed boundary cuts through foreground; review crop boundaries."])
    }
    return CharacterPanelDetection(sourceSHA256: sourceSHA256, sourceOrientation: sourceOrientation,
      detectorVersion: detectorVersion, candidates: candidates,
      status: missingGutterEvidence ? .needsReview : .detected,
      diagnostics: missingGutterEvidence ? ["One or more panel gutters lack measured low-edge evidence; review crop boundaries."] : [])
  }
}
