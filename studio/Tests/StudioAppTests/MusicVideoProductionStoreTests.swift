import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class MusicVideoProductionStoreTests: XCTestCase {
  @MainActor func testRunningProductionPublishesShotProgressBeforeRenderCompletes() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let entered = expectation(description: "render entered")
    let progress = expectation(description: "running shot published")
    var finish: CheckedContinuation<[String: Any], Error>?
    let projectID = UUID(), clipID = UUID()
    let store = StudioStore(dataDirectory: directory, restoreSession: false, invocation: { command, _, body, _ in
      XCTAssertEqual(body["jobDirectory"] as? String, "/test/live")
      if command == "production-run" {
        return try await withCheckedThrowingContinuation { finish = $0; entered.fulfill() }
      }
      XCTAssertEqual(command, "production-status")
      return ["jobDirectory": "/test/live", "projectID": projectID.uuidString, "status": "running",
              "units": [["id": clipID.uuidString, "name": "Second shot", "clipIDs": [clipID.uuidString],
                         "status": "running", "attempts": 1]]]
    })
    store.project.id = projectID
    store.project.production = MusicVideoProduction(jobDirectory: "/test/live",
      inputFingerprint: try store.project.productionInputFingerprint(),
      executionFingerprint: try store.productionExecutionFingerprint())
    let observation = store.$productionStatus.sink { value in
      if value?.status == "running" { progress.fulfill() }
    }
    let run = Task { await store.runProduction() }
    await fulfillment(of: [entered, progress], timeout: 3)
    XCTAssertTrue(store.productionRunning)
    XCTAssertEqual(store.productionStatus?.units.first?.attempts, 1)
    XCTAssertEqual(store.productionStatus?.units.first?.status, "running")
    finish?.resume(returning: ["jobDirectory": "/test/live", "projectID": projectID.uuidString,
                              "status": "completed", "units": []])
    await run.value
    XCTAssertEqual(store.productionStatus?.status, "completed")
    XCTAssertFalse(store.productionRunning)
    withExtendedLifetime(observation) {}
  }

  @MainActor func testLateProgressCannotOverwriteCompletedProduction() async throws {
    try await checkLateProgress(replaceDocument: false)
  }
  @MainActor func testLateProgressCannotAttachToReplacedDocument() async throws {
    try await checkLateProgress(replaceDocument: true)
  }
  @MainActor private func checkLateProgress(replaceDocument: Bool) async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let entered = expectation(description: "render entered")
    let polling = expectation(description: "status entered")
    var finish: CheckedContinuation<[String: Any], Error>?
    var status: CheckedContinuation<[String: Any], Error>?
    let projectID = UUID()
    let store = StudioStore(dataDirectory: directory, restoreSession: false, invocation: { command, _, _, _ in
      if command == "production-run" {
        return try await withCheckedThrowingContinuation { finish = $0; entered.fulfill() }
      }
      XCTAssertEqual(command, "production-status")
      return try await withCheckedThrowingContinuation { status = $0; polling.fulfill() }
    })
    store.project.id = projectID
    store.project.production = MusicVideoProduction(jobDirectory: "/test/live",
      inputFingerprint: try store.project.productionInputFingerprint(),
      executionFingerprint: try store.productionExecutionFingerprint())
    let run = Task { await store.runProduction() }
    await fulfillment(of: [entered, polling], timeout: 3)
    if replaceDocument {
      store.project = StudioProject()
      store.productionStatus = nil
    }
    finish?.resume(returning: ["jobDirectory": "/test/live", "projectID": projectID.uuidString,
                              "status": "completed", "units": []])
    await run.value
    status?.resume(returning: ["jobDirectory": "/test/live", "projectID": projectID.uuidString,
                              "status": "running", "units": []])
    for _ in 0..<10 { await Task.yield() }
    XCTAssertEqual(store.productionStatus?.status, replaceDocument ? nil : "completed")
    XCTAssertFalse(store.productionRunning)
  }

  @MainActor private func checkReopenedTake(resolvedFingerprint: String, shouldGenerate: Bool) async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let movie = directory.appendingPathComponent("existing.mp4")
    try Data().write(to: movie)
    var commands: [String] = []
    var generateIDs: [String]?
    var projectID = UUID()
    let store = StudioStore(dataDirectory: directory, restoreSession: false, invocation: { command, _, body, _ in
      commands.append(command)
      if command == "describe-generation" {
        return ["fingerprint": resolvedFingerprint, "sourcePaths": [], "readinessErrors": []]
      }
      generateIDs = body["generateIDs"] as? [String]
      return ["jobDirectory": "/test/job", "projectID": projectID.uuidString, "status": "ready", "units": []]
    })
    var clip = Clip(name: "Previously rendered")
    clip.engine = .ltx25
    clip.sourcePath = movie.path
    store.project.clips = [clip]
    projectID = store.project.id
    store.generationDescriptions[clip.id] = ["studioInput": store.generationRequestKey(for: clip),
                                             "fingerprint": "original", "sourcePaths": []]
    store.project.clips[0].renderedSignature = store.signature(for: clip)
    // Reopening clears these process-local descriptions while retaining the selected take.
    store.generationDescriptions.removeAll()
    await store.createProduction(maxRetries: 1, allowRemote: false)
    XCTAssertEqual(commands, ["describe-generation", "production-create"])
    XCTAssertEqual(generateIDs, shouldGenerate ? [clip.id.uuidString] : [])
    XCTAssertNil(store.error)
  }
  @MainActor func testReopenedMatchingTakeIsRevalidatedBeforeChoosingGenerationIDs() async throws {
    try await checkReopenedTake(resolvedFingerprint: "original", shouldGenerate: false)
  }
  @MainActor func testReopenedTakeWithChangedResolvedInputsStillGenerates() async throws {
    try await checkReopenedTake(resolvedFingerprint: "changed-model", shouldGenerate: true)
  }
  @MainActor func testEditDuringTakeRevalidationDoesNotCreateProduction() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let movie = directory.appendingPathComponent("existing.mp4")
    try Data().write(to: movie)
    var commands: [String] = []
    var changeInput: (() -> Void)?
    let store = StudioStore(dataDirectory: directory, restoreSession: false, invocation: { command, _, _, _ in
      commands.append(command)
      changeInput?()
      return ["fingerprint": "original", "sourcePaths": [], "readinessErrors": []]
    })
    var clip = Clip(name: "Previously rendered")
    clip.engine = .ltx25
    clip.sourcePath = movie.path
    clip.renderedSignature = "saved-before-reopening"
    store.project.clips = [clip]
    changeInput = { store.project.clips[0].prompt = "Changed while validation was running" }
    await store.createProduction(maxRetries: 1, allowRemote: false)
    XCTAssertEqual(commands, ["describe-generation"])
    XCTAssertNil(store.project.production)
    XCTAssertNil(store.generationDescriptions[clip.id])
    XCTAssertTrue(store.error?.contains("movie changed") == true)
  }
  @MainActor func testExecutionIdentityIncludesRuntimeGlobalAssetsAndGroups() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let store = StudioStore(dataDirectory: directory, restoreSession: false)
    let original = try store.productionExecutionFingerprint()
    store.runtime.profilesDirectory = "/changed/profiles"
    XCTAssertNotEqual(try store.productionExecutionFingerprint(), original)
    let runtimeOnly = try store.productionExecutionFingerprint()
    store.globalAssets.append(MediaAsset(name: "Reference", kind: .image, path: "/changed.png"))
    XCTAssertNotEqual(try store.productionExecutionFingerprint(), runtimeOnly)
  }
  @MainActor func testLateProductionCreationDoesNotAttachToReplacedMovie() async throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let entered = expectation(description: "create entered")
    var resume: CheckedContinuation<[String: Any], Error>?
    let store = StudioStore(dataDirectory: directory, restoreSession: false, invocation: { _, _, _, _ in
      try await withCheckedThrowingContinuation { resume = $0; entered.fulfill() }
    })
    store.project.clips = [Clip(name: "Shot")]
    let id = store.project.id
    let task = Task { await store.createProduction(maxRetries: 1, allowRemote: false) }
    await fulfillment(of: [entered], timeout: 2)
    store.project = StudioProject()
    resume?.resume(returning: ["jobDirectory": "/test/job", "projectID": id.uuidString, "status": "ready", "units": []])
    await task.value
    XCTAssertNil(store.project.production)
    XCTAssertNil(store.productionStatus)
  }
}
