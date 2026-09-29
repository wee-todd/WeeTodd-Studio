import Foundation

/// Sparse payloads keep header-validation tests independent of model downloads.
public func withTensorFile<T>(metadata: [String: String] = [:],
  tensors: [(String, [Int], String)], scalars: [String: Float] = [:], payloads: [String: Data] = [:],
  _ body: (URL) throws -> T) throws -> T {
  let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
  var header: [String: Any] = ["__metadata__": metadata]
  var offset = 0
  var scalarOffsets: [String: Int] = [:]
  for (name, shape, dtype) in tensors {
    let width = ["F32": 4, "F16": 2, "BF16": 2, "F64": 8, "I8": 1, "U8": 1, "U32": 4][dtype]!
    let count = shape.reduce(1, *) * width
    header[name] = ["dtype": dtype, "shape": shape, "data_offsets": [offset, offset + count]]
    scalarOffsets[name] = offset
    offset += count
  }
  let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
  var length = UInt64(json.count).littleEndian
  var data = withUnsafeBytes(of: &length) { Data($0) }
  data.append(json)
  try data.write(to: url)
  defer { try? FileManager.default.removeItem(at: url) }
  let handle = try FileHandle(forWritingTo: url)
  defer { try? handle.close() }
  try handle.truncate(atOffset: UInt64(data.count + offset))
  for (name, value) in scalars {
    var bits = value.bitPattern.littleEndian
    try handle.seek(toOffset: UInt64(data.count + scalarOffsets[name]!))
    try handle.write(contentsOf: withUnsafeBytes(of: &bits) { Data($0) })
  }
  for (name, bytes) in payloads {
    try handle.seek(toOffset: UInt64(data.count + scalarOffsets[name]!))
    try handle.write(contentsOf: bytes)
  }
  return try body(url)
}
