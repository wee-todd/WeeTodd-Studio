import Foundation
import StudioCore

/// Keep bounded header reads and folder enumeration off the UI actor. Native
/// library inspection does not need an installed worker or model tensor payload.
enum NativeLoRAImport {
  static func usesNative(runtime: RuntimeSettings, modelHint: LoRAModel? = nil,
    selectedEngine: Engine? = nil) -> Bool {
    let engine = modelHint.map { Engine(rawValue: $0.rawValue)! } ?? selectedEngine
    if engine == .h3 && runtime.usesNativeH3 { return true }
    if (engine == .ltx23 || engine == .ltx25) && runtime.usesNativeLTX25 { return true }
    return !FileManager.default.isExecutableFile(atPath: runtime.pythonPath)
  }
  static func usesNativeScan(runtime: RuntimeSettings) -> Bool {
    runtime.usesNativeH3 || runtime.usesNativeLTX25 || !FileManager.default.isExecutableFile(atPath: runtime.pythonPath)
  }
  static func inspect(_ url: URL, modelHint: LoRAModel?, profile: String? = nil) async throws -> [String: Any] {
    try await Task.detached(priority: .utility) { try NativeLoRAInspection.inspect(url, modelHint: modelHint, selectedH3Profile: profile) }.value
  }
  static func scan(_ folders: [LoRAFolder]) async throws -> [String: Any] {
    try await Task.detached(priority: .utility) { try NativeLoRAFolderScan.scan(folders) }.value
  }
}
