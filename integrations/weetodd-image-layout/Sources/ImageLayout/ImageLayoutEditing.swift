import Foundation

public enum LayoutResizeHandle: String, CaseIterable, Sendable {
  case topLeft, top, topRight, right, bottomRight, bottom, bottomLeft, left
  public var changesLeft: Bool { [.topLeft, .left, .bottomLeft].contains(self) }
  public var changesRight: Bool { [.topRight, .right, .bottomRight].contains(self) }
  public var changesTop: Bool { [.topLeft, .top, .topRight].contains(self) }
  public var changesBottom: Bool { [.bottomLeft, .bottom, .bottomRight].contains(self) }
}

public enum ImageLayoutEditing {
  public static func descendants(of id: UUID, in draft: ImageLayoutDraft) -> Set<UUID> {
    var ids: Set<UUID> = [id]
    for _ in draft.regions.indices {
      let before = ids.count
      for region in draft.regions where region.parentID.map({ ids.contains($0) }) == true { ids.insert(region.id) }
      if before == ids.count { break }
    }
    return ids
  }
  public static func validateParents(_ draft: ImageLayoutDraft) throws {
    let parents = Dictionary(draft.regions.map { ($0.id, $0.parentID) }, uniquingKeysWith: { first, _ in first })
    for region in draft.regions {
      var seen: Set<UUID> = [region.id], next = region.parentID
      while let id = next {
        guard parents.keys.contains(id), seen.insert(id).inserted else { throw ImageLayoutError.invalid("Layout groups contain a missing parent or cycle.") }
        next = parents[id] ?? nil
      }
    }
  }
  public static func replacing(_ replacement: ImageLayoutRegion, in draft: ImageLayoutDraft) throws -> ImageLayoutDraft {
    try validateParents(draft)
    guard let index = draft.regions.firstIndex(where: { $0.id == replacement.id }) else { throw ImageLayoutError.invalid("Select an existing region.") }
    var result = draft
    let old = draft.regions[index].bounds, new = replacement.bounds
    try ImageLayoutAnalysisGeometry.validate(old)
    try ImageLayoutAnalysisGeometry.validate(new)
    let group = descendants(of: replacement.id, in: draft)
    for i in result.regions.indices where group.contains(result.regions[i].id) && i != index {
      let box = draft.regions[i].bounds
      result.regions[i].bounds = .init(x: new.x + (box.x - old.x) * new.width / old.width,
        y: new.y + (box.y - old.y) * new.height / old.height,
        width: box.width * new.width / old.width, height: box.height * new.height / old.height)
      try ImageLayoutAnalysisGeometry.validate(result.regions[i].bounds)
    }
    result.regions[index] = replacement
    try validateParents(result)
    return result
  }
  public static func move(_ rect: NormalizedLayoutRect, dx: Double, dy: Double) -> NormalizedLayoutRect {
    .init(x: max(0, min(1 - rect.width, rect.x + dx)), y: max(0, min(1 - rect.height, rect.y + dy)), width: rect.width, height: rect.height)
  }
  public static func resize(_ rect: NormalizedLayoutRect, handle: LayoutResizeHandle, dx: Double, dy: Double,
    minimumWidth: Double, minimumHeight: Double) -> NormalizedLayoutRect {
    var left = rect.x, top = rect.y, right = rect.x + rect.width, bottom = rect.y + rect.height
    if handle.changesLeft { left = max(0, min(right - minimumWidth, left + dx)) }
    if handle.changesRight { right = max(left + minimumWidth, min(1, right + dx)) }
    if handle.changesTop { top = max(0, min(bottom - minimumHeight, top + dy)) }
    if handle.changesBottom { bottom = max(top + minimumHeight, min(1, bottom + dy)) }
    return .init(x: left, y: top, width: right - left, height: bottom - top)
  }
}

public struct ImageLayoutHistory {
  private struct Entry { var before, after: ImageLayoutDraft }
  private var past: [Entry] = [], future: [Entry] = []
  private var observedAnalysis = Set<String>()
  public var canUndo: Bool { !past.isEmpty }
  public var canRedo: Bool { !future.isEmpty }
  public init() {}
  /// External proposals arrive through the parent binding. Undo/redo observing
  /// the same request again must not insert another history entry.
  public mutating func observeAnalysisReplacement(from before: ImageLayoutDraft, to after: ImageLayoutDraft) {
    if let id = before.analysis?.requestID { observedAnalysis.insert(id) }
    guard let id = after.analysis?.requestID, before.analysis?.requestID != id,
      observedAnalysis.insert(id).inserted else { return }
    record(before: before, after: after)
  }
  public mutating func record(before: ImageLayoutDraft, after: ImageLayoutDraft) {
    guard before != after else { return }
    past.append(.init(before: before, after: after)); if past.count > 100 { past.removeFirst() }; future = []
  }
  public mutating func undo(current: ImageLayoutDraft) -> ImageLayoutDraft? {
    guard let entry = past.popLast() else { return nil }; future.append(entry)
    return applying(from: entry.after, to: entry.before, current: current)
  }
  public mutating func redo(current: ImageLayoutDraft) -> ImageLayoutDraft? {
    guard let entry = future.popLast() else { return nil }; past.append(entry)
    return applying(from: entry.before, to: entry.after, current: current)
  }
  private func applying(from: ImageLayoutDraft, to: ImageLayoutDraft, current: ImageLayoutDraft) -> ImageLayoutDraft {
    func changed<T: Equatable>(_ before: T, _ after: T, _ now: T) -> T { before != after && now == before ? after : now }
    var result = current
    result.width = changed(from.width, to.width, current.width); result.height = changed(from.height, to.height, current.height)
    result.summary.caption = changed(from.summary.caption, to.summary.caption, current.summary.caption)
    result.summary.meta = changed(from.summary.meta, to.summary.meta, current.summary.meta)
    result.summary.canvas = changed(from.summary.canvas, to.summary.canvas, current.summary.canvas)
    result.summary.palette = changed(from.summary.palette, to.summary.palette, current.summary.palette)
    result.regions = changed(from.regions, to.regions, current.regions)
    result.guideImagePath = changed(from.guideImagePath, to.guideImagePath, current.guideImagePath)
    result.guideOpacity = changed(from.guideOpacity, to.guideOpacity, current.guideOpacity)
    result.analysis = changed(from.analysis, to.analysis, current.analysis)
    result.sourceReconstruction = changed(from.sourceReconstruction, to.sourceReconstruction, current.sourceReconstruction)
    return result
  }
}
