import NNC

/// Explicit developer qualification policies. Float32 remains the default.
/// Mixed policies require explicit opt-in and independent trajectory/media gates.
public enum LTXPrecisionPolicy: String, Codable, CaseIterable, Sendable {
  case float32
  case float16Projections = "fp16-projections"
  case bfloat16Projections = "bf16-projections"

  var projectionType: DataType {
    switch self {
    case .float32: return .Float32
    case .float16Projections: return .Float16
    case .bfloat16Projections: return .BFloat16
    }
  }
}
