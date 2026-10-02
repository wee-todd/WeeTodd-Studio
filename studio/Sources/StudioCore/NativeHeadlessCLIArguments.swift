import Foundation

/// Native jobs accept only controls implemented by their host. Legacy jobs keep
/// the existing Python parser, rather than silently losing its extra options.
public struct NativeHeadlessCLIArguments {
  public let jobPath: String
  public let outputDirectory: String
  public let workerOverrides: [String: String]
  public let resume: Bool
  public let preflightOnly: Bool
  public init(_ arguments: [String]) throws {
    let valued: Set<String> = ["--job", "--output-directory", "--h3-swift-worker", "--ltx25-swift-worker"]
    let switches: Set<String> = ["--resume", "--preflight-only"]
    var values: [String: String] = [:], flags = Set<String>(), index = 0
    while index < arguments.count {
      let flag = arguments[index]
      if valued.contains(flag) {
        guard values[flag] == nil, index + 1 < arguments.count,
          !arguments[index + 1].isEmpty, !arguments[index + 1].hasPrefix("--") else {
          throw StudioError.invalid("Native job option \(flag) needs one value and cannot be repeated.")
        }
        values[flag] = arguments[index + 1]; index += 2
      } else if switches.contains(flag) {
        guard flags.insert(flag).inserted else { throw StudioError.invalid("Native job option \(flag) cannot be repeated.") }
        index += 1
      } else { throw StudioError.invalid("Native jobs do not support the option \(flag). No settings were applied.") }
    }
    guard let job = values["--job"], let output = values["--output-directory"] else {
      throw StudioError.invalid("Pass --job and --output-directory for a native job.")
    }
    jobPath = job; outputDirectory = output; resume = flags.contains("--resume")
    preflightOnly = flags.contains("--preflight-only")
    var overrides: [String: String] = [:]
    if let value = values["--h3-swift-worker"] { overrides["h3"] = value }
    if let value = values["--ltx25-swift-worker"] { overrides["ltx25"] = value }
    workerOverrides = overrides
  }
}
