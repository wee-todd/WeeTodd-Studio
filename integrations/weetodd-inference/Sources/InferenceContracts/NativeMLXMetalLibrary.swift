import Foundation

/// The worker and cache receipt must bind the same actual Metal library.
/// Packaged workers never fall back to a shadow file beside the executable.
public enum NativeMLXMetalLibrary {
  public static func resolve(executable:URL)throws->URL {
    guard executable.isFileURL,!executable.path.utf8.contains(0) else { throw ContractError.invalid("MLX requires a local worker executable.") }
    let resolved=executable.standardizedFileURL.resolvingSymlinksInPath()
    let folder=resolved.deletingLastPathComponent()
    let selected=folder.lastPathComponent == "MacOS"
      ? folder.deletingLastPathComponent().appendingPathComponent("Resources/LTXNative/mlx.metallib")
      : folder.appendingPathComponent("mlx.metallib")
    let library=selected.standardizedFileURL.resolvingSymlinksInPath()
    guard (try? library.resourceValues(forKeys:[.isRegularFileKey]).isRegularFile) == true else {
      throw ContractError.invalid("The selected native MLX Metal library is missing or is not a regular file: \(selected.path)")
    }
    return library
  }
}
