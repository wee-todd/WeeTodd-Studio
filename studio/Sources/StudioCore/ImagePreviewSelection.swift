import Foundation

/// Immutable selection: file identity is independent of the displayed name or list position.
public struct ImagePreviewSelection: Identifiable, Equatable, Sendable {
  public let path: String
  public let title: String
  public var id: String { path }

  public init(path: String, title: String? = nil) {
    let url = URL(fileURLWithPath: path).standardizedFileURL
    self.path = url.path
    self.title = title ?? url.lastPathComponent
  }

  public static func references(paths: [String], subject: String) -> [Self] {
    var seen = Set<String>()
    var items: [Self] = []
    for path in paths {
      let item = Self(path: path, title: "\(subject) · Reference \(items.count + 1)")
      if seen.insert(item.id).inserted { items.append(item) }
    }
    return items
  }
}
