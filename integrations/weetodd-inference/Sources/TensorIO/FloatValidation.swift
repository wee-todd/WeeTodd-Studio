import Foundation

/// Bitwise finite classification, including signaling NaNs. No floating-point
/// reduction may suppress a NaN or overflow while checking large finite values.
public enum FloatValidation {
  public static func allFinite(_ values: [Float]) -> Bool {
    values.withUnsafeBytes { bytes in
      let mask = SIMD16<UInt32>(repeating: 0x7f800000)
      var index = 0
      while index + 16 <= values.count {
        let bits = bytes.loadUnaligned(fromByteOffset: index * 4, as: SIMD16<UInt32>.self)
        if (bits & mask).max() == 0x7f800000 { return false }
        index += 16
      }
      while index < values.count {
        if !values[index].isFinite { return false }
        index += 1
      }
      return true
    }
  }
}
