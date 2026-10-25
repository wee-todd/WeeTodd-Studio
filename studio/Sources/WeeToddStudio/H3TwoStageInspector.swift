import AppKit
import StudioCore
import SwiftUI

struct H3TwoStageInspector: View {
  @EnvironmentObject var store: StudioStore
  var clip: Clip
  private var selected: H3TwoStageSettings? {clip.generationSelection?.h3Joint?.twoStage}
  private func edit(_ body:@escaping(inout H3TwoStageSettings)->Void) {
    store.editClip { clip in
      guard var settings=clip.generationSelection?.h3Joint?.twoStage else {return}
      body(&settings);clip.generationSelection?.h3Joint?.twoStage=settings
    }
  }
  private func chooseTurbo() {
    let panel=NSOpenPanel();panel.canChooseDirectories=false;panel.allowsMultipleSelection=false
    guard panel.runModal() == .OK,let url=panel.url else {return}
    let expected=selected,session=store.documentSessionID,projectID=store.project.id,task=clip.inferredTask,engine=clip.engine
    Task {
      do {
        let path=try await Task.detached {
          let info=try NativeLoRAInspection.validateH3Sampling(path:url.path,selectedProfile:"turbo",schedulePoints:5)
          guard info["loraLayout"] as? String != "native_interleaved" else {throw StudioError.invalid("Choose a contiguous-QKV four-step H3 Turbo component for this default preset.")}
          return url.standardizedFileURL.resolvingSymlinksInPath().path
        }.value
        guard store.documentSessionID==session,store.project.id==projectID,store.selectedClip?.id==clip.id,store.selectedClip?.engine==engine,store.selectedClip?.inferredTask==task,store.selectedClip?.generationSelection?.h3Joint?.twoStage==expected else {return}
        edit {$0.draftTurboPath=path;$0.draftTurboTask=H3TwoStageSettings.turboTask(for:task)}
        var links=store.h3TwoStageModels
        links.setTurboPath(path,task:task)
        store.saveH3TwoStageModels(links)
      } catch {store.error=error.localizedDescription}
    }
  }
  var body: some View {
    DisclosureGroup("Two-stage upscale") {
      Toggle("Generate a smaller draft, then upscale",isOn:Binding(get:{selected != nil},set:{enabled in
        store.editClip {clip in
          var selection=clip.generationSelection ?? GenerationSelection(task:clip.inferredTask,preset:.custom)
          var joint=selection.h3Joint ?? H3JointSettings()
          joint.twoStage=enabled ? store.h3TwoStageModels.settings(task:clip.inferredTask) : nil
          if enabled {joint.refinement=nil}
          selection.h3Joint = !joint.saveFullLatents && joint.refinement==nil && joint.twoStage==nil ? nil : joint
          clip.generationSelection=selection
        }
      }))
      if let selected {
        Picker("Draft sampling",selection:Binding(get:{selected.useTurboDraft},set:{value in edit {$0.useTurboDraft=value}})) {
          Text("Turbo · 4 Euler evaluations").tag(true)
          Text("Current sampler and LoRA settings").tag(false)
        }
        if selected.useTurboDraft {
          Text(selected.draftTurboPath.map {"Draft Turbo: \(URL(fileURLWithPath:$0).lastPathComponent) · strength 1"} ?? "Link a Turbo adapter, or use the profile's existing strength-1 Turbo.").font(.caption).foregroundStyle(.secondary)
          Button("Choose draft Turbo adapter…") {chooseTurbo()}
        }
        Text("Output width and height set the exact final resolution.").font(.caption).foregroundStyle(.secondary)
        Picker("Draft scale",selection:Binding(get:{selected.draftScale},set:{value in edit {$0.draftScale=value}})) {
          Text("1.5×").tag(1.5);Text("2×").tag(2.0)
        }
        if let canvas=try? selected.draftCanvas(width:clip.generationWidth,height:clip.generationHeight,durationSeconds:clip.duration) {
          Text("Draft \(canvas.width) × \(canvas.height) → final \(clip.generationWidth) × \(clip.generationHeight)").font(.caption).textSelection(.enabled)
        } else {
          Text("Choose a valid 32-pixel final canvas and a duration that fits H3. Longest edge ≤1920, shortest edge ≤1088; references also consume the row budget.").font(.caption).foregroundStyle(.orange)
        }
        Stepper("Refinement evaluations: \(selected.evaluations)",value:Binding(get:{selected.evaluations},set:{value in edit {$0.evaluations=value}}),in:1...64)
        if selected.resizeMethod==nil {
          Text(selected.learnedUpscalerPath.map {"Learned: \(URL(fileURLWithPath:$0).lastPathComponent)"} ?? "Link the learned H3 latent upscaler before generation.").font(.caption).textSelection(.enabled)
          Button("Use interpolation") {edit {$0.learnedUpscalerPath=nil;$0.learnedUpscalerHeaderSHA256=nil;$0.resizeMethod = .bilinear}}
        } else {
          Picker("Latent resize",selection:Binding(get:{selected.resizeMethod ?? .bilinear},set:{value in edit {$0.resizeMethod=value}})) {
            ForEach(H3JointResizeMethod.allCases,id: \.self) {Text($0.rawValue).tag($0)}
          }
        }
        Button("Choose learned H3 latent upscaler…") {
          let panel=NSOpenPanel();panel.canChooseDirectories=false;panel.allowsMultipleSelection=false
          guard panel.runModal() == .OK,let url=panel.url else {return}
          let expected=selected,session=store.documentSessionID,projectID=store.project.id,task=clip.inferredTask,engine=clip.engine
          Task {
            do {
              let model=try await Task.detached {try NativeH3LearnedUpscalerMetadata.inspect(path:url.path)}.value
              guard store.documentSessionID==session,store.project.id==projectID,store.selectedClip?.id==clip.id,store.selectedClip?.engine==engine,store.selectedClip?.inferredTask==task,store.selectedClip?.generationSelection?.h3Joint?.twoStage==expected else {return}
              edit {$0.learnedUpscalerPath=model.path;$0.learnedUpscalerHeaderSHA256=model.headerSHA256;$0.resizeMethod=nil}
              var links=store.h3TwoStageModels;links.learnedUpscalerPath=model.path;links.learnedUpscalerHeaderSHA256=model.headerSHA256
              store.saveH3TwoStageModels(links)
            } catch {store.error=error.localizedDescription}
          }
        }
        Text("The final pass preserves draft audio. Identified Turbo adapters apply to the draft; standard creative LoRAs apply to both passes. Strength and evaluation count are independent of draft steps.").font(.caption2).foregroundStyle(.secondary)
      }
    }
    .onChange(of:clip.inferredTask) {task in
      guard store.selectedClip?.id==clip.id,let stage=selected,stage.useTurboDraft,
        stage.draftTurboTask != H3TwoStageSettings.turboTask(for:task) else {return}
      let linked=store.h3TwoStageModels.settings(task:task)
      edit {$0.draftTurboPath=linked.draftTurboPath;$0.draftTurboTask=linked.draftTurboTask}
    }
  }
}
