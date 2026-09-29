import Accelerate
import Darwin

public enum Float16Conversion {
  /// Native vector conversion with explicit nearest-even rounding. Reject
  /// nonrepresentable weights before conversion; do not alter thread rounding.
  public static func nearest(_ values: [Float]) throws -> [Float16] {
    guard fegetround() == FE_TONEAREST else {
      throw CheckpointError.invalid("Float16 weights require nearest-even rounding mode.")
    }
    let valid = values.withUnsafeBytes { bytes in
      let mask = SIMD16<UInt32>(repeating: 0x7fffffff)
      var index = 0
      while index+16 <= values.count {
        let bits = bytes.loadUnaligned(fromByteOffset: index*4,as: SIMD16<UInt32>.self)
        if (bits & mask).max() > 0x477fe000 { return false }
        index += 16
      }
      return values[index...].allSatisfy { ($0.bitPattern & 0x7fffffff) <= 0x477fe000 }
    }
    guard valid else { throw CheckpointError.invalid("Weight exceeds finite Float16 range.") }
    guard !values.isEmpty else { return [] }
    try Task.checkCancellation()
    var result = [Float16](repeating: 0,count: values.count)
    let status = values.withUnsafeBytes { input in
      result.withUnsafeMutableBytes { output in
        var src = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: input.baseAddress!),height: 1,
          width: vImagePixelCount(values.count),rowBytes: values.count*4)
        var dst = vImage_Buffer(data: output.baseAddress!,height: 1,
          width: vImagePixelCount(values.count),rowBytes: values.count*2)
        return vImageConvert_PlanarFtoPlanar16F(&src,&dst,vImage_Flags(kvImageDoNotTile))
      }
    }
    guard status == kvImageNoError else { throw CheckpointError.invalid("Native Float16 conversion failed: \(status)") }
    try Task.checkCancellation()
    return result
  }
}
