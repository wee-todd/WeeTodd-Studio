import Foundation
public struct FlashVSRCLIArguments {
  public let requestPath: String, outputDirectory: String
  public let preflightOnly: Bool, resume: Bool
  public init(_ arguments: [String]) throws {
    var values: [String: String] = [:], flags = Set<String>(), index = 0
    while index < arguments.count {
      let flag = arguments[index]
      if ["--request", "--output-directory"].contains(flag) {
        guard values[flag] == nil, index + 1 < arguments.count, !arguments[index + 1].isEmpty,
          !arguments[index + 1].hasPrefix("--") else { throw StudioError.invalid("Upscale option " + flag + " needs one value and cannot repeat.") }
        values[flag] = arguments[index + 1]; index += 2
      } else if ["--preflight-only", "--resume"].contains(flag), flags.insert(flag).inserted { index += 1 }
      else { throw StudioError.invalid("Unknown or repeated upscale option: " + flag) }
    }
    guard let request = values["--request"], let output = values["--output-directory"] else {
      throw StudioError.invalid("Pass --request and --output-directory for standalone upscaling.")
    }
    requestPath = request; outputDirectory = output
    preflightOnly = flags.contains("--preflight-only"); resume = flags.contains("--resume")
  }
}
