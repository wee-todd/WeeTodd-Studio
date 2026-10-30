import Foundation

public struct NativeUpscalerSetupArguments: Sendable {
  public enum Action: String, Sendable { case list, download, convert, sharpTable = "sharp-table",prepareH3 = "prepare-h3" }
  public let action: Action
  public let values: [String: String]
  public let sources: [String]
  public init(_ arguments: [String]) throws {
    guard let first = arguments.first, let action = Action(rawValue: first), arguments.count <= 64 else {
      throw StudioError.invalid("Use models list, download, convert, sharp-table or prepare-h3.")
    }
    var values: [String: String] = [:], sources: [String] = [], i = 1
    let allowed: Set<String>
    switch action {
    case .list: allowed = ["--catalog"]
    case .download: allowed = ["--catalog", "--package", "--model-library", "--reuse-root"]
    case .convert: allowed = ["--profile", "--source", "--model-library"]
    case .sharpTable: allowed = ["--source", "--model-library"]
    case .prepareH3: allowed = ["--profile","--source","--model-library","--worker","--catalog","--profile-directory","--include-fast"]
    }
    while i < arguments.count {
      let key = arguments[i]
      guard allowed.contains(key), i + 1 < arguments.count, !arguments[i + 1].hasPrefix("--"),
        !arguments[i + 1].isEmpty, arguments[i + 1].utf8.count <= 4096, !arguments[i + 1].utf8.contains(0) else {
        throw StudioError.invalid("Invalid or unsupported native model setup argument.")
      }
      let value = arguments[i + 1]
      if key == "--reuse-root" || (key == "--source" && [.sharpTable,.prepareH3].contains(action)) { sources.append(value) }
      else {
        guard values[key] == nil else { throw StudioError.invalid("Duplicate native setup argument: " + key) }
        values[key] = value
      }
      i += 2
    }
    let required: [String]
    switch action {
    case .list: required = []
    case .download: required = ["--package", "--model-library"]
    case .convert: required = ["--profile", "--source", "--model-library"]
    case .sharpTable: required = ["--model-library"]
    case .prepareH3: required = ["--model-library"]
    }
    guard required.allSatisfy({ values[$0] != nil }) else { throw StudioError.invalid("Required native model setup arguments are missing.") }
    if action == .prepareH3 {
      let profile=values["--profile"] ?? "restoration"
      guard ["restoration","ref2va-base","publisher-encoder"].contains(profile),
        values["--include-fast"].map({["true","false"].contains($0)}) ?? true,
        profile == "restoration" ? sources.isEmpty : ["--catalog","--profile-directory","--include-fast"].allSatisfy({values[$0]==nil}),
        profile != "ref2va-base" || sources.count <= 1,
        profile != "publisher-encoder" || [0,2].contains(sources.count) else {
        throw StudioError.invalid("Invalid H3 preparation profile, source count or profile-specific option.")
      }
    }
    self.action = action; self.values = values; self.sources = sources
  }
}
