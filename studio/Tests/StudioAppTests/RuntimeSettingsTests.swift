import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class RuntimeSettingsTests: XCTestCase {
  func testVoiceModelsAreRuntimeOnlyAndLegacySettingsStillDecode() throws {
    var settings = RuntimeSettings(root: "/root", pythonPath: "/python", profilesDirectory: "/profiles")
    XCTAssertNil(try JSONDecoder().decode(RuntimeSettings.self, from: JSONEncoder().encode(settings)).voiceModels)
    var models = VoiceModelSettings()
    let model = models.register(InstalledVoiceModel(engine: .fishS2Pro, name: "Fish S2 Pro · 8-bit", path: "/models/fish"))
    settings.voiceModels = models
    let restored = try JSONDecoder().decode(RuntimeSettings.self, from: JSONEncoder().encode(settings))
    XCTAssertEqual(restored.voiceModels?.preferred(for: .fishS2Pro)?.id, model.id)
    XCTAssertNil(settings.generationSettings.voiceModels)
  }
  private let defaults = RuntimeSettings(
    root: "/new/source", pythonPath: "/new/python", profilesDirectory: "/new/profiles",
    drawThingsHelperPath: "/Applications/Studio.app/Contents/MacOS/WeeToddDrawThings")

  func testLegacySettingsAdoptBundledHelperWithoutReplacingNativeRuntime() throws {
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let bundled = try executable(at: directory.appendingPathComponent("bundled-helper"))
    var currentDefaults = defaults; currentDefaults.drawThingsHelperPath = bundled.path
    let legacy = Data("""
      {"root":"/existing/source","pythonPath":"/existing/python",
       "profilesDirectory":"/existing/profiles","ffmpegPath":"/existing/ffmpeg",
       "ffprobePath":"","rifePath":"","rifeWeights":"","metalPath":""}
      """.utf8)
    let result = RuntimeSettings.restoring(legacy, defaults: currentDefaults)
    XCTAssertEqual(result.drawThingsHelperPath, bundled.path)
    XCTAssertEqual(result.root, "/existing/source")
    XCTAssertEqual(result.pythonPath, "/existing/python")
    XCTAssertEqual(result.ffmpegPath, "/existing/ffmpeg")
  }

  func testValidExecutableCustomHelperIsPreserved() throws {
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let custom = try executable(at: directory.appendingPathComponent("custom-helper"))
    let bundled = try executable(at: directory.appendingPathComponent("bundled-helper"))
    let saved = RuntimeSettings(root: "/saved/root", pythonPath: "/saved/python", profilesDirectory: "/saved/profiles",
      drawThingsHelperPath: custom.path)
    var currentDefaults = defaults; currentDefaults.drawThingsHelperPath = bundled.path
    let result = RuntimeSettings.restoring(try JSONEncoder().encode(saved), defaults: currentDefaults)
    XCTAssertEqual(result.drawThingsHelperPath, custom.path)
    XCTAssertEqual(result.root, saved.root)
    XCTAssertEqual(result.pythonPath, saved.pythonPath)
  }

  func testEmptyHelperUsesBundleAndMissingBundleRemainsOptional() throws {
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let bundled = try executable(at: directory.appendingPathComponent("bundled-helper"))
    var currentDefaults = defaults; currentDefaults.drawThingsHelperPath = bundled.path
    var saved = currentDefaults
    saved.drawThingsHelperPath = ""
    let data = try JSONEncoder().encode(saved)
    XCTAssertEqual(RuntimeSettings.restoring(data, defaults: currentDefaults).drawThingsHelperPath, bundled.path)
    var withoutHelper = currentDefaults
    withoutHelper.drawThingsHelperPath = nil
    XCTAssertNil(RuntimeSettings.restoring(data, defaults: withoutHelper).drawThingsHelperPath)
  }

  func testStaleSavedHelperFallsBackToExecutableBundledDefaultWithoutReplacingRuntime() throws {
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let bundled = try executable(at: directory.appendingPathComponent("bundled-helper"))
    var saved = RuntimeSettings(root: "/saved/root", pythonPath: "/saved/python", profilesDirectory: "/saved/profiles")
    saved.drawThingsHelperPath = directory.appendingPathComponent("deleted-helper").path
    var currentDefaults = defaults; currentDefaults.drawThingsHelperPath = bundled.path
    let result = RuntimeSettings.restoring(try JSONEncoder().encode(saved), defaults: currentDefaults)
    XCTAssertEqual(result.drawThingsHelperPath, bundled.path)
    XCTAssertEqual(result.root, "/saved/root")
    XCTAssertEqual(result.pythonPath, "/saved/python")
    XCTAssertEqual(result.profilesDirectory, "/saved/profiles")
  }

  func testNonExecutableSavedHelperFallsBackButMissingBundledDefaultDoesNotInventPath() throws {
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let nonExecutable = directory.appendingPathComponent("non-executable")
    try Data("helper".utf8).write(to: nonExecutable)
    let bundled = try executable(at: directory.appendingPathComponent("bundled-helper"))
    var saved = defaults; saved.drawThingsHelperPath = nonExecutable.path
    var currentDefaults = defaults; currentDefaults.drawThingsHelperPath = bundled.path
    XCTAssertEqual(RuntimeSettings.restoring(try JSONEncoder().encode(saved), defaults: currentDefaults).drawThingsHelperPath,
                   bundled.path)
    currentDefaults.drawThingsHelperPath = directory.appendingPathComponent("missing-bundled-helper").path
    XCTAssertNil(RuntimeSettings.restoring(try JSONEncoder().encode(saved), defaults: currentDefaults).drawThingsHelperPath)
  }

  private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("runtime-settings-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }

  private func executable(at url: URL) throws -> URL {
    try Data("#!/bin/sh\nexit 0\n".utf8).write(to: url)
    try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
    return url
  }
}

extension RuntimeSettingsTests {
  func testAccelerationSettingsRemainOptionalForLegacyRuntime() throws {
    var runtime = RuntimeSettings(root: "/source", pythonPath: "/python", profilesDirectory: "/profiles")
    XCTAssertNil(try JSONDecoder().decode(RuntimeSettings.self,
      from: JSONEncoder().encode(runtime)).acceleration)
    runtime.acceleration = .init()
    runtime.acceleration?.h3MemoryPolicy = "paged"
    let restored = RuntimeSettings.restoring(try JSONEncoder().encode(runtime), defaults: runtime)
    XCTAssertEqual(restored.acceleration?.h3MemoryPolicy, "paged")
    XCTAssertEqual(restored.acceleration?.h3ProjectionBackend, "auto")
  }
}

extension RuntimeSettingsTests {
  func testLargerWorkspacePagingRuntimeRoundTrip() throws {
    var runtime = RuntimeSettings(root: "/source", pythonPath: "/python", profilesDirectory: "/profiles")
    runtime.acceleration = .init()
    runtime.acceleration?.h3MemoryPolicy = "pagedNormal"
    let restored = RuntimeSettings.restoring(try JSONEncoder().encode(runtime), defaults: runtime)
    XCTAssertEqual(restored.acceleration?.h3MemoryPolicy, "pagedNormal")
    XCTAssertEqual(restored.acceleration?.h3ProjectionBackend, "auto")
  }
}
