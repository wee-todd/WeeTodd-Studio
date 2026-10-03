import AVFoundation
import Foundation
import XCTest
@testable import StudioCore

final class NativeHeadlessRippleTests: XCTestCase {
  private let ffmpeg = "/opt/homebrew/bin/ffmpeg"
  private func root() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("NativeRippleJob-\(UUID())")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }; return root
  }
  private func movie(_ root: URL, name: String, duration: Double, audio: Bool) throws -> URL {
    guard FileManager.default.isExecutableFile(atPath: ffmpeg) else { throw XCTSkip("Tiny FFmpeg fixture required") }
    let url = root.appendingPathComponent(name + ".mp4"), p = Process()
    p.executableURL = URL(fileURLWithPath: ffmpeg)
    var args = ["-v", "error", "-f", "lavfi", "-i", "testsrc2=s=64x64:r=24:d=\(duration)"]
    if audio { args += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=\(duration)"] }
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio { args += ["-c:a", "aac", "-ac", "2"] }
    p.arguments = args + [url.path]; p.standardError = FileHandle.nullDevice
    try p.run(); p.waitUntilExit(); XCTAssertEqual(p.terminationStatus, 0); return url
  }
  private func fixture(_ root: URL, silent: Bool = false) throws -> (NativeHeadlessJob, Data, RippleDraft, URL) {
    let source = try movie(root, name: "source", duration: 2, audio: true)
    let output = try movie(root, name: "fixture", duration: 1, audio: !silent)
    var project = StudioProject(); project.settings.width = 64; project.settings.height = 64
    var clip = Clip(engine: .movie); clip.sourcePath = source.path; clip.sourceIn = 0.5; clip.duration = 1
    var draft = RippleDraft(clip: clip, frameRate: 24); draft.width = 64; draft.height = 64
    draft.seed = 429; draft.prompt = "Frozen edit"; draft.audioPolicy = silent ? .silent : .preserve
    draft.references[0].path = root.appendingPathComponent("first.png").path; draft.references[0].strength = 0.61
    var anchor = RippleReference(frame: 12, path: root.appendingPathComponent("anchor.png").path); anchor.strength = 0.37
    draft.references.append(anchor)
    for reference in draft.references { try Data("edited-\(reference.frame)".utf8).write(to: URL(fileURLWithPath: reference.path)) }
    clip.rippleDraft = draft; project.clips = [clip]
    let guide = root.appendingPathComponent("guide.rgb"); try Data(repeating: 4, count: 25 * 64 * 64 * 3).write(to: guide)
    let target = root.appendingPathComponent("frozen-take")
    let raw: [String: Any] = ["version": 1, "engine": "ltx25", "task": "ripple",
      "gemma_root": "/components/gemma", "transformer_root": "/components/transformer", "connector_checkpoint": "/components/connector",
      "video_checkpoint": "/components/video", "audio_checkpoint": "/components/audio", "adapter_path": "/components/adapter",
      "adapter_strength": draft.loraStrength, "guide_path": guide.path, "first_reference_path": draft.references[0].path,
      "source_path": source.path, "source_sha256": try NativeHeadlessJob.fileHash(source), "source_start": 0.5,
      "duration": 1, "editorial_frames": 24, "width": 64, "height": 64, "frames": 25, "fps": 24, "seed": draft.seed,
      "prompt": draft.prompt, "reference_strength": 0.61, "anchors": [["frame": 12, "path": draft.references[1].path, "strength": 0.37]],
      "audio_policy": draft.audioPolicy.rawValue, "ffmpeg_path": ffmpeg, "output_directory": target.path]
    let bytes = try JSONSerialization.data(withJSONObject: raw, options: [.sortedKeys])
    let report = try JSONSerialization.data(withJSONObject: ["rippleDraft": JSONSerialization.jsonObject(with: JSONEncoder().encode(draft))])
    let job = try NativeHeadlessJob(project: project, recipes: [clip.id.uuidString: .init(engine: "ltx25", bytes: bytes,
      signature: draft.inputFingerprint, report: report)], workers: ["ltx25": "/usr/bin/true"], ffmpeg: ffmpeg)
    return (job, bytes, draft, output)
  }
  private func publish(bytes: Data, draft: RippleDraft, media: URL, target: URL, fault: String? = nil) throws -> [String: Any] {
    try FileManager.default.createDirectory(at: target, withIntermediateDirectories: false)
    let movie = target.appendingPathComponent("ripple.mp4"); try FileManager.default.copyItem(at: media, to: movie)
    var refs: [[String: Any]] = []
    for ref in draft.references {
      let copy = target.appendingPathComponent("ref-\(ref.frame).png"); try FileManager.default.copyItem(atPath: ref.path, toPath: copy.path)
      refs.append(["frame": ref.frame, "path": copy.path, "strength": Double(Float(ref.strength))])
    }
    let hash = try NativeHeadlessJob.fileHash(URL(fileURLWithPath: draft.sourcePath))
    var result: [String: Any] = ["jobID": UUID().uuidString, "nativeRuntime": "swift-mlx", "video_path": movie.path,
      "path": movie.path, "duration": 1.0, "frames": 24, "frame_rate": 24.0, "width": 64, "height": 64,
      "has_audio": draft.audioPolicy != .silent, "receipt_path": target.appendingPathComponent("receipt.json").path,
      "artifacts_directory": target.path, "frozen_references": refs, "source_sha256": hash]
    var receipt: [String: Any] = ["format": "weetodd-ripple-take-v1", "status": "complete", "source_sha256": hash,
      "source_path": draft.sourcePath, "source_start": draft.sourceIn, "duration": draft.duration,
      "editorial_frames": 24, "reference_images": refs, "publication_audio": draft.audioPolicy == .silent ? "silent" : "preserved source interval"]
    if fault == "foreign" { receipt["source_sha256"] = String(repeating: "0", count: 64) }
    if fault == "missing" { result.removeValue(forKey: "source_sha256") }
    if fault == "escaped" { result["video_path"] = media.path }
    try bytes.write(to: target.appendingPathComponent("ripple-request.json"))
    try JSONSerialization.data(withJSONObject: receipt).write(to: target.appendingPathComponent("receipt.json"))
    try JSONSerialization.data(withJSONObject: result).write(to: target.appendingPathComponent("result.json"))
    try Data("{}".utf8).write(to: target.appendingPathComponent("report.json"))
    return result
  }
  func testRipplePreservesDedicatedRequestSilentOrSourceAudioTakeAcceptanceAndResume() async throws {
    for silent in [false, true] {
      let root = try root(), (job, bytes, draft, media) = try fixture(root, silent: silent)
      let target = root.appendingPathComponent("frozen-take"), output = root.appendingPathComponent("CLI")
      var calls: [String] = []
      let result = try await NativeHeadlessExecutor.run(job: job, output: output, emit: { _ in }, worker: { _, recipe, destination, mode in
        calls.append(mode); XCTAssertEqual(try Data(contentsOf: recipe), bytes); XCTAssertEqual(destination, target)
        if mode == "preflight" { return [:] }
        return try self.publish(bytes: bytes, draft: draft, media: media, target: destination)
      })
      XCTAssertEqual(calls, ["preflight", "render"]); XCTAssertEqual(result["newlyGenerated"] as? Int, 1)
      let reopened = try ProjectStorage.read(output.appendingPathComponent("result.weetodd")), clip = reopened.clips[0]
      let take = try XCTUnwrap(clip.rippleTakes?.first)
      XCTAssertEqual(take.submittedDraftFingerprint, draft.inputFingerprint); XCTAssertEqual(take.draft.sourceIn, 0.5)
      XCTAssertEqual(take.draft.seed, 429); XCTAssertEqual(take.draft.loraStrength, draft.loraStrength)
      XCTAssertEqual(take.hasAudio, !silent); XCTAssertEqual(take.draft.references.map(\.strength), draft.references.map(\.strength))
      XCTAssertEqual(clip.sourcePath, take.path); XCTAssertEqual(clip.sourceIn, 0); XCTAssertEqual(clip.duration, 1)
      XCTAssertEqual(clip.versions[0].path, draft.sourcePath); XCTAssertEqual(clip.versions[0].usableSourceIn, 0.5)
      XCTAssertEqual(clip.versions.last?.recipePath, take.receiptPath); XCTAssertEqual(clip.reviewedTakeFingerprint, clip.reuseFingerprint)
      XCTAssertEqual(reopened.assets[0].width, 64)
      let before = try NativeHeadlessJob.fileHash(URL(fileURLWithPath: take.path))
      let resumed = try await NativeHeadlessExecutor.run(job: job, output: output, resume: true, emit: { _ in }, worker: { _, _, _, _ in
        XCTFail("Ripple resume must use its accepted take without a worker"); return [:]
      })
      XCTAssertEqual(resumed["newlyGenerated"] as? Int, 0); XCTAssertEqual(resumed["resumedGenerations"] as? Int, 1)
      XCTAssertEqual(try NativeHeadlessJob.fileHash(URL(fileURLWithPath: take.path)), before)
      let stateURL = output.appendingPathComponent("job-state.json"), stateBytes = try Data(contentsOf: stateURL)
      var state = try JSONSerialization.jsonObject(with: stateBytes) as! [String: Any]
      state.removeValue(forKey: "rippleArtifactSHA256")
      try JSONSerialization.data(withJSONObject: state).write(to: stateURL)
      do { _ = try await NativeHeadlessExecutor.run(job: job, output: output, resume: true, emit: { _ in }); XCTFail("Missing replay identities must invalidate resume") } catch { }
      try stateBytes.write(to: stateURL)
      try Data("tampered receipt".utf8).write(to: URL(fileURLWithPath: take.receiptPath))
      do { _ = try await NativeHeadlessExecutor.run(job: job, output: output, resume: true, emit: { _ in }); XCTFail("Changed Ripple receipts must invalidate resume") } catch { }
    }
  }
  func testDedicatedRippleRequestUsesActualWorkerEnvelopeAndVideoPathReceipt() async throws {
    let root = try root(), (original, bytes, draft, media) = try fixture(root, silent: true)
    let target = root.appendingPathComponent("frozen-take")
    _ = try publish(bytes: bytes, draft: draft, media: media, target: target)
    let template = root.appendingPathComponent("template"); try FileManager.default.moveItem(at: target, to: template)
    let executable = root.appendingPathComponent("worker")
    let script = #"""
    #!/bin/sh
    set -eu
    mode="$1"; request="$3"; output="$5"
    id=$(/usr/bin/sed -n 's/.*"jobID"[ ]*:[ ]*"\([^"]*\)".*/\1/p' "$request")
    printf '{"event":"progress","stage":"fixture","fraction":0.5}\n'
    if [ "$mode" = render ]; then
      /bin/cp -R '\#(template.path)' "$output"
      /usr/bin/sed "s/\"jobID\"[ ]*:[ ]*\"[^\"]*\"/\"jobID\":\"$id\"/" "$output/result.json" > "$output/identity.json"
      /bin/mv "$output/identity.json" "$output/result.json"
      printf '{"status":"success","result":'
      /bin/cat "$output/result.json"
      printf '}\n'
    else
      printf '{"status":"success","result":{"jobID":"%s","nativeRuntime":"swift-mlx","task":"ripple"}}\n' "$id"
    fi
    """#
    try script.write(to: executable, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: executable.path)
    var job = original; job.workers["ltx25"] = executable.path
    let result = try await NativeHeadlessExecutor.run(job: job, output: root.appendingPathComponent("CLI"), emit: { _ in })
    XCTAssertEqual(result["newlyGenerated"] as? Int, 1)
    XCTAssertEqual(try Data(contentsOf: target.appendingPathComponent("ripple-request.json")), bytes)
    XCTAssertFalse(FileManager.default.fileExists(atPath: target.appendingPathComponent("video.mp4").path))
  }
  func testRippleRejectsForeignMissingAndEscapedPublicationIdentities() async throws {
    for fault in ["foreign", "missing", "escaped"] {
      let root = try root(), (job, bytes, draft, media) = try fixture(root)
      let output = root.appendingPathComponent("CLI")
      do {
        _ = try await NativeHeadlessExecutor.run(job: job, output: output, emit: { _ in }, worker: { _, _, target, mode in
          if mode == "preflight" { return [:] }; return try self.publish(bytes: bytes, draft: draft, media: media, target: target, fault: fault)
        }); XCTFail("Invalid Ripple publication must be rejected: \(fault)")
      } catch { }
      XCTAssertFalse(FileManager.default.fileExists(atPath: output.appendingPathComponent("result.weetodd").path))
      XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("frozen-take").path))
    }
  }
  func testRippleFrozenSourceDigestDetectsEqualSizeRestoredTimestampMutationBeforeWorker() async throws {
    let root = try root(), (job, _, draft, _) = try fixture(root)
    let url = URL(fileURLWithPath: draft.sourcePath), attributes = try FileManager.default.attributesOfItem(atPath: url.path)
    var bytes = try Data(contentsOf: url); bytes[bytes.count - 1] ^= 1; try bytes.write(to: url)
    try FileManager.default.setAttributes([.modificationDate: attributes[.modificationDate]!], ofItemAtPath: url.path)
    var calls = 0
    do { _ = try await NativeHeadlessExecutor.run(job: job, output: root.appendingPathComponent("CLI"), worker: { _, _, _, _ in calls += 1; return [:] }); XCTFail("Changed source identity must fail") } catch { }
    XCTAssertEqual(calls, 0)
  }
  func testRippleAdmissionRejectsMissingDraftChangedSettingsAndUnknownFields() throws {
    let root = try root(), (job, bytes, _, _) = try fixture(root)
    let id = job.project.clips[0].id.uuidString
    for changed in ["seed", "version", "audio_policy", "unexpected", "report"] {
      var modified = job
      if changed == "report" { modified.recipes[id]!.report = Data() }
      else {
        var raw = try JSONSerialization.jsonObject(with: bytes) as! [String: Any]
        if changed == "version" { raw[changed] = true }
        else if changed == "audio_policy" { raw[changed] = "silent" }
        else { raw[changed] = 99 }
        modified.recipes[id]!.bytes = try JSONSerialization.data(withJSONObject: raw)
      }
      XCTAssertThrowsError(try modified.validate(), "Reject changed Ripple admission: \(changed)")
    }
    var changed = job
    changed.recipes[id]!.bytes = try JSONSerialization.data(withJSONObject: ["format": "weetodd-headless-v2", "engine": "ltx25"])
    changed.recipes[id]!.report = Data()
    changed.project.clips[0].engine = .ltx25
    XCTAssertThrowsError(try changed.validate(), "Pending Ripple cannot be admitted as ordinary LTX")
    let manifest = root.appendingPathComponent("ripple.json"); try job.write(to: manifest)
    XCTAssertEqual(try NativeHeadlessJob.read(from: manifest).recipes[id]!.bytes, bytes)
  }
  func testRippleCancelledAfterWorkerCannotAcceptOrOverwriteExistingTake() async throws {
    let root = try root(), (job, _, _, _) = try fixture(root), token = NativeHeadlessCancellation()
    do {
      _ = try await NativeHeadlessExecutor.run(job: job, output: root.appendingPathComponent("CLI"), cancellation: token, emit: { _ in }, worker: { _, _, target, mode in
        if mode == "preflight" { return [:] }
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: false)
        token.cancel(); throw CancellationError()
      }); XCTFail("Cancelled Ripple must not be accepted")
    } catch { XCTAssertTrue(error is CancellationError) }
    XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("frozen-take").path))
    let target = root.appendingPathComponent("frozen-take"); try FileManager.default.createDirectory(at: target, withIntermediateDirectories: false)
    let marker = target.appendingPathComponent("existing"); try Data("preserve".utf8).write(to: marker)
    do { _ = try await NativeHeadlessExecutor.run(job: job, output: root.appendingPathComponent("other"), emit: { _ in }, worker: { _, _, _, _ in [:] }); XCTFail("Existing frozen output must fail") } catch { }
    XCTAssertTrue(FileManager.default.fileExists(atPath: marker.path))
  }
}
