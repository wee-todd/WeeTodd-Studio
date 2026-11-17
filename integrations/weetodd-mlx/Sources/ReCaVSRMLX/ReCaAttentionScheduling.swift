import Foundation

public enum ReCaAttentionScheduling: String, Sendable {
  case automatic = "auto", synchronous = "sync"
  func usesAsync(architecture: String) -> Bool { self == .automatic && architecture == "applegpu_g15d" }
}
