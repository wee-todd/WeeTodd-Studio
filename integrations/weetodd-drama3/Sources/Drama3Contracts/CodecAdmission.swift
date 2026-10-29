import Foundation

/// Validate the entire fixed codec architecture without allocating model arrays.
public enum Drama3CodecAdmission {
  public static func inspect(_ file: URL) throws {
    try validate(Drama3TensorFile(file).tensors)
  }
  static func validate(_ tensors: [String: Drama3TensorInfo]) throws {
    let expected = Drama3CodecTensorSchema.shapes
    guard Set(tensors.keys) == Set(expected.keys) else {
      throw Drama3Error.invalid(
        "The codec checkpoint is incomplete or uses an unsupported architecture. Restore the verified codec.safetensors file."
      )
    }
    for (name, shape) in expected {
      guard let tensor = tensors[name], tensor.shape == shape,
        ["F32", "F16", "BF16"].contains(tensor.dtype)
      else {
        throw Drama3Error.invalid(
          "Incompatible codec tensor: \(name). Restore the verified codec checkpoint.")
      }
    }
  }
}
