#!/usr/bin/env swift
import CryptoKit
import Foundation

// Development build/distribution entry point. Runtime model setup and inference
// are supplied by the native worker and StudioCore, without an interpreter.
func option(_ name: String) -> String? {
  guard let i = CommandLine.arguments.firstIndex(of: name), i + 1 < CommandLine.arguments.count else { return nil }
  return CommandLine.arguments[i + 1]
}
func run(_ command: [String], capture: Bool = false) throws -> String {
  let task = Process(); task.executableURL = URL(fileURLWithPath: command[0]); task.arguments = Array(command.dropFirst())
  let pipe = Pipe(); if capture { task.standardOutput = pipe }
  try task.run()
  let bytes = capture ? pipe.fileHandleForReading.readDataToEndOfFile() : Data()
  task.waitUntilExit()
  guard task.terminationStatus == 0 else { throw NSError(domain: "MOSS build", code: Int(task.terminationStatus), userInfo: [NSLocalizedDescriptionKey: "Build command failed: \(command.joined(separator: " "))"]) }
  return String(decoding: bytes, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
}
func sha(_ url: URL) throws -> String {
  let stream = try FileHandle(forReadingFrom: url); defer { try? stream.close() }; var hash = SHA256()
  while let data = try stream.read(upToCount: 1024 * 1024), !data.isEmpty { hash.update(data: data) }
  return hash.finalize().map { String(format: "%02x", $0) }.joined()
}
do {
  let fm = FileManager.default
  let root = URL(fileURLWithPath: option("--root") ?? URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().path)
  let package = root.appendingPathComponent("integrations/weetodd-moss-mlx")
  let configuration = option("--configuration") ?? "release"
  guard ["debug", "release"].contains(configuration) else { throw NSError(domain: "MOSS build", code: 1) }
  let build = ["/usr/bin/xcrun", "swift", "build", "--package-path", package.path, "-c", configuration, "--build-system", "native"]
  _ = try run(build + ["--product", "WeeToddMOSSImageWorker", "--jobs", "2", "--disable-automatic-resolution", "-Xcxx", "-DNDEBUG"])
  let binaries = URL(fileURLWithPath: try run(build + ["--show-bin-path"], capture: true))
  let checkouts = package.appendingPathComponent(".build/checkouts")
  _ = try run(["/bin/bash", root.appendingPathComponent("integrations/weetodd-mlx/scripts/build_mlx_metallib.sh").path,
    binaries.path, checkouts.appendingPathComponent("mlx-swift").path])
  let output = URL(fileURLWithPath: option("--output") ?? package.appendingPathComponent(".build/worker-distribution").path)
  try fm.createDirectory(at: output, withIntermediateDirectories: true)
  for name in ["WeeToddMOSSImageWorker", "mlx.metallib"] {
    let target = output.appendingPathComponent(name)
    if fm.fileExists(atPath: target.path) { try fm.removeItem(at: target) }
    try fm.copyItem(at: binaries.appendingPathComponent(name), to: target)
  }
  let resolved = try JSONSerialization.jsonObject(with: Data(contentsOf: package.appendingPathComponent("Package.resolved"))) as! [String: Any]
  let pins = resolved["pins"] as! [[String: Any]]
  var notices = "WeeTodd MOSS-Image native worker. Model weights are installed separately.\n"
  notices += "Independent implementation researched from OpenMOSS MOSS-Image (Apache-2.0), MFLUX (MIT), and published ER-SDE equations (Cui et al., arXiv:2309.06169). ComfyUI and Draw Things are research references; no GPL source is bundled in this worker.\n"
  for pin in pins {
    let identity = pin["identity"] as! String, folder = checkouts.appendingPathComponent(identity)
    guard let iterator = fm.enumerator(at: folder, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey], options: [.skipsHiddenFiles]) else { throw NSError(domain: "Missing dependency licenses", code: 1) }
    var found = false
    for case let file as URL in iterator {
      let values = try file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
      if values.isRegularFile == true, (values.fileSize ?? Int.max) < 1024 * 1024,
        ["LICENSE", "NOTICE", "COPYING"].contains(where: { file.lastPathComponent.uppercased().hasPrefix($0) }) {
        notices += "\n--- \(identity)/\(file.path.replacingOccurrences(of: folder.path + "/", with: "")) ---\n" + (try String(contentsOf: file, encoding: .utf8)); found = true
      }
    }
    guard found else { throw NSError(domain: "Missing licenses for \(identity)", code: 1) }
  }
  try Data(notices.utf8).write(to: output.appendingPathComponent("Notices.txt"), options: .atomic)
  var hashes: [String: String] = [:]
  for name in ["WeeToddMOSSImageWorker", "mlx.metallib", "Notices.txt"] { hashes[name] = try sha(output.appendingPathComponent(name)) }
  let manifest: [String: Any] = ["format": "weetodd-moss-worker-v1", "protocol": 1, "files": hashes, "dependencies": pins]
  try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys]).write(to: output.appendingPathComponent("manifest.json"), options: .atomic)
  print(output.path)
} catch {
  FileHandle.standardError.write(Data((error.localizedDescription + "\n").utf8)); exit(1)
}
