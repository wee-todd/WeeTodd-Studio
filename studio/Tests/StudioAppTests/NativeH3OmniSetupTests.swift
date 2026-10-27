import AppKit
import SwiftUI
import Foundation
import StudioCore
import XCTest
@testable import WeeToddStudio

final class NativeH3OmniSetupTests:XCTestCase {
  @MainActor func testOmniDoesNotAppearInImageModelsAndRetainsSavedImageChoiceAsUnavailable() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at:root) }
    let store=StudioStore(dataDirectory:root,restoreSession:false)
    let profile=try JSONDecoder().decode(ModelProfile.self,from:Data(#"{"id":"omni","name":"Omni","engine":"h3","task":"ref2va","generation":{"supportedTasks":["ref2va"],"fasth3Omni":true,"controls":{"stepsEditable":false,"refinementStepsEditable":false,"cfgEditable":false,"shiftEditable":false,"stepsExplanation":"fixed","cfgExplanation":"fixed","shiftExplanation":"fixed"},"presets":[]}}"#.utf8))
    store.profiles=[profile]
    XCTAssertTrue(store.h3ImageProfiles.isEmpty)
  }
  @MainActor func testOmniSetupSelectsEightEvaluationsWithoutDeletingRetainedWork() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:root) }
    let file=root.appendingPathComponent("omni.json");try Data("{}".utf8).write(to:file)
    let store=StudioStore(dataDirectory:root,restoreSession:false)
    var clip=Clip(engine:.h3);clip.generationSelection = .init(task:"ref2va")
    clip.generationSelection!.h3Joint = .init(twoStage:.init());clip.prompt="Keep me"
    let attachment=Attachment(assetID:UUID(),role:.reference);clip.attachments=[attachment]
    store.project.clips=[clip];store.selectedClipID=clip.id
    let state=ModelSetupState();state.begin(try XCTUnwrap(NativeModelSetup.catalog().first { $0.id==NativeH3OmniProfile.presetID }))
    state.resultPath=file.path;state.useRecipeForSelectedClip(store:store)
    let selected=try XCTUnwrap(store.selectedClip)
    XCTAssertEqual(selected.generationSelection?.steps,8)
    XCTAssertNotNil(selected.generationSelection?.h3Joint)
    XCTAssertEqual(selected.prompt,"Keep me");XCTAssertEqual(selected.attachments,[attachment])
    XCTAssertFalse(NativeH3OmniProfile.conflicts(clip:selected).isEmpty)
    var cleared=selected;NativeH3OmniProfile.applyRequiredSettings(to:&cleared)
    XCTAssertTrue(NativeH3OmniProfile.conflicts(clip:cleared).isEmpty)
    XCTAssertEqual(cleared.prompt,selected.prompt);XCTAssertEqual(cleared.attachments,selected.attachments)
    XCTAssertEqual(cleared.generationWidth,selected.generationWidth)
  }
  @MainActor func testOmniInspectorReportsEightDespiteRetainedTurboDraft() throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at:root) }
    let store=StudioStore(dataDirectory:root,restoreSession:false)
    store.runtime.nativeH3Enabled=true;store.runtime.h3WorkerPath="/usr/bin/true"
    var clip=Clip(engine:.h3);clip.profileID="omni";clip.generationSelection = .init(task:"ref2va")
    clip.generationSelection!.h3Joint = .init(twoStage:.init())
    let generation:[String:Any] = ["supportedTasks":["ref2va"],"fasth3Omni":true,
      "controls":["evaluations":8,"stepsEditable":false,"refinementStepsEditable":false,"cfgEditable":false,"shiftEditable":false,
        "stepsExplanation":"8 actual evaluations","cfgExplanation":"CFG1","shiftExplanation":"fixed"],"presets":[]]
    let descriptor=try JSONDecoder().decode(GenerationDescriptor.self,from:JSONSerialization.data(withJSONObject:generation))
    store.profiles=[ModelProfile(id:"omni",name:"Omni",engine:"h3",task:"ref2va",generation:descriptor)]
    store.project.clips=[clip];store.selectedClipID=clip.id
    store.generationDescriptions[clip.id]=["studioEngine":"h3","studioTask":"ref2va","studioProfile":"omni","generation":generation]
    _=NSApplication.shared
    let host=NSHostingView(rootView:GenerationInspector(clip:clip,detail:.basic).environmentObject(store).frame(width:420,height:1200))
    let window=NSWindow(contentRect:NSRect(x:0,y:0,width:420,height:1200),styleMask:[.borderless],backing:.buffered,defer:false)
    window.isReleasedWhenClosed=false;window.contentView=host;defer {window.close()}
    host.layoutSubtreeIfNeeded();RunLoop.main.run(until:Date().addingTimeInterval(0.08));host.layoutSubtreeIfNeeded()
    var steps:NSTextField?
    func inspect(_ view:NSView) {
      if let field=view as? NSTextField,field.placeholderString=="Steps" {steps=field}
      for child in view.subviews {inspect(child)}
    }
    inspect(host)
    XCTAssertEqual(try XCTUnwrap(steps).integerValue,8)
    XCTAssertFalse(try XCTUnwrap(steps).isEnabled)
    XCTAssertNotNil(store.selectedClip?.generationSelection?.h3Joint)
    XCTAssertFalse(window.isVisible)
  }

}
