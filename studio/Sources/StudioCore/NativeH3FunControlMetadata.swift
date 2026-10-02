import Darwin
import Foundation

/// Weight-free editor admission for full-width Fun Union control profiles.
/// The Swift H3 worker owns complete tensor/offset validation and execution.
enum NativeH3FunControlMetadata {
  private static func header(_ path: String) throws -> [String: Any] {
    let descriptor = Darwin.open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK)
    guard descriptor >= 0 else { throw StudioError.invalid("Relink the H3 Fun control checkpoint or base transformer.") }
    let file = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
    defer { try? file.close() }
    var before = stat()
    guard fstat(descriptor, &before) == 0, before.st_mode & S_IFMT == S_IFREG,
      let prefix = try file.read(upToCount: 8), prefix.count == 8 else {
      throw StudioError.invalid("H3 Fun control requires regular SafeTensors checkpoints.")
    }
    let length = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (8 * $1.offset) }
    guard length > 1, length <= 4 * 1024 * 1024, UInt64(before.st_size) >= length + 8,
      let bytes = try file.read(upToCount: Int(length)), bytes.count == Int(length),
      let value = try JSONSerialization.jsonObject(with: bytes) as? [String: Any] else {
      throw StudioError.invalid("Invalid bounded H3 Fun checkpoint header.")
    }
    var after = stat()
    guard fstat(descriptor, &after) == 0,
      before.st_size == after.st_size,
      before.st_mtimespec.tv_sec == after.st_mtimespec.tv_sec,
      before.st_mtimespec.tv_nsec == after.st_mtimespec.tv_nsec,
      before.st_ctimespec.tv_sec == after.st_ctimespec.tv_sec,
      before.st_ctimespec.tv_nsec == after.st_ctimespec.tv_nsec else {
      throw StudioError.invalid("H3 Fun checkpoint changed during preparation.")
    }
    try Task.checkCancellation()
    return value
  }

  static func validate(control: String, transformer: String) throws {
    let values = try header(control)
    let roots = ["model.diffusion_model.", "diffusion_model.", "controlnet.", ""]
      .filter { values[$0 + "control_proj_in.weight"] != nil }
    guard roots.count == 1 else { throw StudioError.invalid("Select a full-width H3 Fun Union checkpoint.") }
    let root = roots[0]
    func shape(_ object: [String: Any], _ name: String) -> [Int]? {
      (object[name] as? [String: Any])?["shape"] as? [Int]
    }
    let blocks = Set(values.keys.compactMap { name -> Int? in
      guard name.hasPrefix(root + "control_blocks.") else { return nil }
      return Int(name.dropFirst((root + "control_blocks.").count).split(separator: ".").first ?? "")
    })
    guard shape(values, root + "control_proj_in.weight") == [5376, 196],
      blocks == Set(0..<5) || blocks == Set(0..<10),
      blocks.allSatisfy({ shape(values, root + "control_blocks.\($0).adaln_proj.linear.weight") == [96768, 2688] }) else {
      throw StudioError.invalid("Swift H3 Fun control currently needs the full-width 2688-coordinate Union v1 or 2.0 adapter; pruned basis adapters require matching converted weights.")
    }
    let base = try header(transformer)
    let baseRoots = ["model.diffusion_model.", "diffusion_model.", ""]
      .filter { shape(base, $0 + "time_embedder.proj_out.weight") == [2688, 5376] }
    guard baseRoots.count == 1,
      shape(base, baseRoots[0] + "blocks.0.adaln_proj.linear.weight") == [96768, 2688] else {
      throw StudioError.invalid("The H3 base and Fun control adapter must both use full-width 2688-coordinate AdaLN.")
    }
    if let metadata = values["__metadata__"] as? [String: String],
      let places = metadata["control_blocks_places"] {
      let expected = blocks.count == 5 ? [0, 10, 20, 30, 40] : [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
      guard let data = places.data(using: .utf8),
        let declared = try JSONSerialization.jsonObject(with: data) as? [NSNumber],
        declared.count == expected.count,
        zip(declared, expected).allSatisfy({ number, layer in
          CFGetTypeID(number) != CFBooleanGetTypeID() && number.doubleValue == Double(layer)
        }) else {
        throw StudioError.invalid("The H3 Fun adapter declares an unsupported injection schedule.")
      }
    }
  }
}
