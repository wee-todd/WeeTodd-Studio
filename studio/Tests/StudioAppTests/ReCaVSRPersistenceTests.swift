import Foundation
import InferenceContracts
import StudioCore
import XCTest
@testable import WeeToddStudio

final class ReCaVSRPersistenceTests: XCTestCase {
  @MainActor func testChosenModelFolderSurvivesRelaunch() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    let models = directory.appendingPathComponent("Custom ReCa INT8")
    try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
    store.selectReCaVSRModelDirectory(models)
    let reopened = StudioStore(dataDirectory: directory, restoreSession: false)
    XCTAssertEqual(reopened.runtime.reCaVSRModelDirectory, models.path)
  }

  @MainActor func testWholeMoviePreferencesSurviveReopeningAndModelSwitches() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    let settings = try ReCaVSRSettings(scale: 2, seed: 777, precision: .int8ConvRot, colorFix: .adain, residency: .resident)
    store.reCaVSRSettings = settings; store.videoUpscaleTool = .reCaVSR
    store.saveVideoUpscalePreferences()
    let reopened = StudioStore(dataDirectory: directory, restoreSession: false)
    reopened.loadVideoUpscalePreferences()
    XCTAssertEqual(reopened.videoUpscaleTool, .reCaVSR)
    XCTAssertEqual(reopened.reCaVSRSettings, settings)
    reopened.videoUpscaleTool = .flashVSR; reopened.videoUpscaleTool = .reCaVSR
    XCTAssertEqual(reopened.reCaVSRSettings, settings)
  }
}
