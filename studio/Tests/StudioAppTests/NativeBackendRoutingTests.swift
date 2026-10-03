import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class NativeBackendRoutingTests: XCTestCase {
  func testAbsentPreferenceSelectsNativeIndependentOfWorkerAvailability() {
    var runtime = RuntimeSettings(root: "/missing", pythonPath: "/missing/python",
      profilesDirectory: "/missing/profiles")
    XCTAssertNil(runtime.nativeH3Enabled)
    XCTAssertNil(runtime.nativeLTX25Enabled)
    XCTAssertTrue(runtime.usesNativeH3)
    XCTAssertTrue(runtime.usesNativeLTX25)
    runtime.h3WorkerPath = "/missing/h3"
    runtime.ltx25WorkerPath = "/missing/ltx"
    XCTAssertTrue(runtime.usesNativeH3)
    XCTAssertTrue(runtime.usesNativeLTX25)
  }

  func testLegacyRestorationPreservesSettingsAndDefaultsToNative() throws {
    let legacy = Data("""
      {"root":"/existing","pythonPath":"/existing/python","profilesDirectory":"/existing/profiles",
       "ffmpegPath":"/existing/ffmpeg","ffprobePath":"","rifePath":"","rifeWeights":"","metalPath":""}
      """.utf8)
    let defaults = RuntimeSettings(root: "/new", pythonPath: "/new/python",
      profilesDirectory: "/new/profiles", ltx25WorkerPath: "/bundled/ltx", h3WorkerPath: "/bundled/h3")
    let restored = RuntimeSettings.restoring(legacy, defaults: defaults)
    XCTAssertEqual(restored.root, "/existing")
    XCTAssertEqual(restored.pythonPath, "/existing/python")
    XCTAssertEqual(restored.profilesDirectory, "/existing/profiles")
    XCTAssertEqual(restored.ffmpegPath, "/existing/ffmpeg")
    XCTAssertEqual(restored.h3WorkerPath, defaults.h3WorkerPath)
    XCTAssertEqual(restored.ltx25WorkerPath, defaults.ltx25WorkerPath)
    XCTAssertNil(restored.nativeH3Enabled)
    XCTAssertNil(restored.nativeLTX25Enabled)
    XCTAssertTrue(restored.usesNativeH3)
    XCTAssertTrue(restored.usesNativeLTX25)
  }

  func testExplicitLegacyOptOutSurvivesRestorationAndRoundTrip() throws {
    var saved = RuntimeSettings(root: "/existing", pythonPath: "/existing/python",
      profilesDirectory: "/profiles")
    saved.nativeH3Enabled = false
    saved.nativeLTX25Enabled = false
    var defaults = saved
    defaults.nativeH3Enabled = nil
    defaults.nativeLTX25Enabled = nil
    defaults.h3WorkerPath = "/bundled/h3"
    defaults.ltx25WorkerPath = "/bundled/ltx"
    let restored = RuntimeSettings.restoring(try JSONEncoder().encode(saved), defaults: defaults)
    XCTAssertEqual(restored.nativeH3Enabled, false)
    XCTAssertEqual(restored.nativeLTX25Enabled, false)
    XCTAssertFalse(restored.usesNativeH3)
    XCTAssertFalse(restored.usesNativeLTX25)
    XCTAssertEqual(try JSONDecoder().decode(RuntimeSettings.self,
      from: JSONEncoder().encode(restored)).nativeH3Enabled, false)
  }

  func testExecutionIdentityContainsResolvedBackendWithoutMutatingPreferences() {
    var runtime = RuntimeSettings(root: "/missing", pythonPath: "/missing/python", profilesDirectory: "/profiles")
    XCTAssertEqual(runtime.generationSettings.nativeH3Enabled, true)
    XCTAssertEqual(runtime.generationSettings.nativeLTX25Enabled, true)
    XCTAssertNil(runtime.nativeH3Enabled)
    XCTAssertNil(runtime.nativeLTX25Enabled)
    runtime.nativeH3Enabled = false
    XCTAssertEqual(runtime.generationSettings.nativeH3Enabled, false)
  }

  @MainActor func testDefaultH3AndLTXTasksDispatchNativeWithoutPythonOrManualEnable() async throws {
    let tasks: [(Engine, String)] = [(.h3, "t2v"), (.h3, "fflf"), (.h3, "ref2va"),
      (.h3, "a2v"), (.ltx25, "t2v")]
    for (engine, task) in tasks {
      let prefix = engine == .h3 ? "h3-native-" : "ltx-native-"
      let directory = try temporaryDirectory()
      defer { try? FileManager.default.removeItem(at: directory) }
      let prepared = directory.appendingPathComponent("prepared")
      try FileManager.default.createDirectory(at: prepared, withIntermediateDirectories: true)
      let recipe = prepared.appendingPathComponent("recipe.json")
      try Data("{}".utf8).write(to: recipe)
      var calls: [String] = []
      let store = StudioStore(dataDirectory: directory, restoreSession: false,
        invocation: { command, runtime, body, _ in
          calls.append(command)
          XCTAssertEqual(runtime.pythonPath, "/unavailable/python")
          if command == prefix + "describe" {
            let project = try JSONDecoder().decode(StudioProject.self,
              from: JSONSerialization.data(withJSONObject: body["project"]!))
            let clipID = try XCTUnwrap(body["clipID"] as? String)
            let clip = try XCTUnwrap(project.clips.first { $0.id.uuidString == clipID })
            XCTAssertEqual(clip.generationSelection?.task, task)
            return ["fingerprint": "native-" + task, "readinessErrors": []]
          }
          if command == prefix + "prepare" {
            return ["recipePath": recipe.path, "prompt": "Frozen prompt", "report": [:]]
          }
          if command == prefix + "preflight" { return [:] }
          if command == prefix + "render" { throw StudioError.invalid("Stopped before inference") }
          throw StudioError.invalid("Unexpected backend: " + command)
        })
      store.runtime = RuntimeSettings(root: "/unavailable", pythonPath: "/unavailable/python",
        profilesDirectory: directory.path, h3WorkerPath: "/bundled/h3")
      store.addClip()
      store.editClip {
        $0.engine = engine; $0.prompt = "Frozen prompt"; $0.seed = 42
        $0.generationSelection = GenerationSelection(task: task)
      }
      await store.prepareSelected()
      XCTAssertNotNil(store.preparedRecipe, "task \(task): \(store.error ?? "")")
      await store.renderPrepared()
      XCTAssertTrue(calls.contains(prefix + "describe"), task)
      XCTAssertTrue(calls.contains(prefix + "prepare"), task)
      XCTAssertTrue(calls.contains(prefix + "preflight"), task)
      XCTAssertTrue(calls.contains(prefix + "render"), task)
      XCTAssertFalse(calls.contains { ["describe-generation", "prepare", "render", "audio-driver"].contains($0) }, task)
      XCTAssertEqual(store.selectedClip?.seed, 42)
      XCTAssertTrue(store.selectedClip?.versions.isEmpty == true)
    }
  }

  @MainActor func testDefaultNativeExportEligibilityAndBackendIdentityRespectOptOut() throws {
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    store.runtime = RuntimeSettings(root: "/unavailable", pythonPath: "/unavailable/python",
      profilesDirectory: directory.path)
    store.addClip(.h3); store.addClip(.ltx25)
    let clips = store.project.clips
    let clip = try XCTUnwrap(store.selectedClip)
    let defaultKey = store.generationRequestKey(for: clip)
    XCTAssertTrue(store.nativeHeadlessEligible)
    store.runtime.nativeH3Enabled = true; store.runtime.nativeLTX25Enabled = true
    XCTAssertEqual(store.generationRequestKey(for: clip), defaultKey,
      "An explicit native preference resolves to the same execution backend")
    store.runtime.nativeLTX25Enabled = false
    XCTAssertNotEqual(store.generationRequestKey(for: clip), defaultKey)
    XCTAssertFalse(store.nativeHeadlessEligible)
    XCTAssertEqual(store.project.clips, clips, "A backend preference must preserve authored clips and takes")
  }

  @MainActor func testMissingNativeWorkersFailBeforePythonProcessOrRecipeRead() async throws {
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let runtime = RuntimeSettings(root: "/unavailable", pythonPath: "/unavailable/python",
      profilesDirectory: directory.path)
    for engine in ["h3", "ltx"] {
      do {
        _ = try await Bridge().invoke(engine + "-native-preflight", runtime: runtime,
          payload: ["recipePath": "/unavailable/recipe"], output: directory.appendingPathComponent(engine))
        XCTFail("Missing worker must fail")
      } catch {
        XCTAssertTrue(error.localizedDescription.contains("worker is missing"), error.localizedDescription)
        XCTAssertFalse(error.localizedDescription.contains("Python"))
      }
    }
  }

  @MainActor func testUnsupportedNativeGenerationDoesNotRetryPython() async throws {
    for engine in [Engine.h3, .ltx25] {
      let directory = try temporaryDirectory()
      defer { try? FileManager.default.removeItem(at: directory) }
      var calls: [String] = []
      let expected = engine == .h3 ? "h3-native-describe" : "ltx-native-describe"
      let store = StudioStore(dataDirectory: directory, restoreSession: false,
        invocation: { command, _, _, _ in
          calls.append(command)
          throw StudioError.invalid(command == expected ? "Unsupported native task" : "Unexpected legacy fallback")
        })
      store.runtime = RuntimeSettings(root: "/unavailable", pythonPath: "/unavailable/python", profilesDirectory: directory.path)
      store.addClip(); store.editClip { $0.engine = engine; $0.prompt = "Frozen prompt" }
      await store.prepareSelected()
      XCTAssertEqual(store.error, "Unsupported native task")
      XCTAssertNil(store.preparedRecipe)
      XCTAssertTrue(calls.contains(expected))
      XCTAssertFalse(calls.contains("describe-generation"))
    }
  }

  @MainActor func testActualNativeProfileAdmissionFailsWithoutPythonFallback() async throws {
    for engine in [Engine.h3, .ltx25] {
      let directory = try temporaryDirectory()
      defer { try? FileManager.default.removeItem(at: directory) }
      // A present JSON profile is not native authority merely because its
      // engine matches. An unsupported component task must remain inadmissible.
      let profile: [String: Any] = ["format": "weetodd-headless-v2",
        "engine": engine.rawValue, "components": ["task": "unsupported"],
        "config": [:], "prompt": "Frozen prompt"]
      try JSONSerialization.data(withJSONObject: profile).write(to: directory.appendingPathComponent("unsupported.json"))
      let store = StudioStore(dataDirectory: directory.appendingPathComponent("store"), restoreSession: false)
      store.runtime = RuntimeSettings(root: "/unavailable", pythonPath: "/unavailable/python",
        profilesDirectory: directory.path)
      store.addClip(); store.editClip { $0.engine = engine; $0.prompt = "Frozen prompt" }
      await store.prepareSelected()
      XCTAssertNil(store.preparedRecipe)
      XCTAssertTrue(store.error?.contains(engine == .h3 ? "Swift H3" : "Swift LTX") == true,
        store.error ?? "Expected native admission error")
      XCTAssertFalse(store.error?.contains("Python environment") == true)
    }
  }

  @MainActor func testExplicitLegacyOptOutDispatchesOnlyRequestedLegacyBackend() async throws {
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    var calls: [String] = []
    let store = StudioStore(dataDirectory: directory, restoreSession: false,
      invocation: { command, _, _, _ in
        calls.append(command)
        throw StudioError.invalid("Explicit legacy selection")
      })
    store.runtime.nativeH3Enabled = false
    store.runtime.nativeLTX25Enabled = false
    store.addClip(); store.editClip { $0.engine = .h3; $0.prompt = "Frozen prompt" }
    await store.prepareSelected()
    XCTAssertTrue(calls.contains("describe-generation"))
    XCTAssertFalse(calls.contains("h3-native-describe"))
    XCTAssertEqual(store.error, "Explicit legacy selection")
  }

  private func temporaryDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("native-default-routing-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
  }
}
