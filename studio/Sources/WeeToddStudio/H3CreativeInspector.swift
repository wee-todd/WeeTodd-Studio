import StudioCore
import SwiftUI

struct H3CreativeInspector:View {
  @EnvironmentObject var store:StudioStore
  var clip:Clip
  private func edit(_ body:@escaping(inout GenerationSelection)->Void) {
    store.editClip { c in
      var s=c.generationSelection ?? GenerationSelection(task:c.inferredTask,preset:.custom)
      body(&s);c.generationSelection=s
    }
  }
  private var sources:[H3JointLatentArtifact] {
    store.project.clips.flatMap { $0.versions.compactMap(\.jointLatentArtifact) }
  }
  var body:some View {
    if ["ref2va","a2v","i2v","fflf","extension"].contains(clip.inferredTask) {
      DisclosureGroup("Reference conditioning") {
        noise("Visual",key: \.visualConditionStrength)
        noise("Audio",key: \.audioConditionStrength)
        Text("Recipe defaults are preserved until an override is enabled.").font(.caption2).foregroundStyle(.secondary)
      }
    }
    if clip.continuityMode == "independent",["t2v","t2va","i2v","fflf","ref2va","a2v"].contains(clip.inferredTask) {
      DisclosureGroup("Full latent refinement · experimental") {
        Toggle("Save complete audio/video latents",isOn:Binding(get:{clip.generationSelection?.h3Joint?.saveFullLatents ?? false},set:{ value in
          edit { s in var joint=s.h3Joint ?? H3JointSettings();joint.saveFullLatents=value
            s.h3Joint = !joint.saveFullLatents && joint.refinement == nil ? nil : joint }
        }))
        Picker("Refinement source",selection:Binding(get:{clip.generationSelection?.h3Joint?.refinement?.source.manifest ?? ""},set:{ path in
          edit { s in var joint=s.h3Joint ?? H3JointSettings()
            joint.refinement=sources.first(where:{$0.manifest==path}).map { H3JointRefinementSettings(mode:.initialized,source:$0) }
            s.h3Joint = !joint.saveFullLatents && joint.refinement == nil ? nil : joint }
        })) {
          Text("No refinement").tag("")
          ForEach(sources,id: \.manifest) { source in
            Text("\(source.task) · \(source.width) × \(source.height) · \(source.generatedFrames) frames").tag(source.manifest)
          }
        }
        if let selected=clip.generationSelection?.h3Joint?.refinement {
          Picker("Mode",selection:Binding(get:{selected.mode.rawValue},set:{ value in
            edit { s in guard let mode=H3JointRefinementMode(rawValue:value) else {return}
              s.h3Joint?.refinement?.mode=mode;s.h3Joint?.refinement?.resizeMethod=mode == .spatial ? .bilinear : nil
              s.h3Joint?.refinement?.learnedUpscalerPath=nil;s.h3Joint?.refinement?.learnedUpscalerHeaderSHA256=nil;s.h3Joint?.refinement?.expandedSpatialTarget=nil }
          })) {
            Text("Same canvas").tag("initialized");Text("Larger canvas").tag("spatial")
          }
          HStack {
            Text("Refinement strength")
            Slider(value:Binding(get:{selected.strength},set:{v in edit {$0.h3Joint?.refinement?.strength=v}}),in:0.01...1)
            Text(selected.strength,format:.number.precision(.fractionLength(2)))
          }
          if selected.mode == .initialized && selected.source.width*selected.source.height>1376*768 {
            Text("This expanded artifact can be stored and reopened. Initialized same-canvas refinement currently keeps the ordinary 1 MP budget.").font(.caption2).foregroundStyle(.orange)
          }
          Toggle("Preserve source audio latents",isOn:Binding(get:{selected.preserveAudio},set:{v in edit {$0.h3Joint?.refinement?.preserveAudio=v}}))
          if selected.mode == .spatial {
            H3LearnedUpscalerInspector(clip:clip,selected:selected)
            if selected.learnedUpscalerPath==nil {
            Picker("Latent resize",selection:Binding(get:{selected.resizeMethod ?? .bilinear},set:{v in edit {$0.h3Joint?.refinement?.resizeMethod=v}})) {
              ForEach(H3JointResizeMethod.allCases,id: \.self) { Text($0.rawValue).tag($0) }
            }
            }
            Text("Increase both output axes by at most 2×. The complete source duration and model components must match.").font(.caption2).foregroundStyle(.secondary)
          }
        }
      }
    }
  }
  private func noise(_ title:String,key:WritableKeyPath<H3ReferenceSettings,Double?>)->some View {
    VStack(alignment:.leading) {
      Toggle("Override \(title.lowercased()) conditioning",isOn:Binding(get:{clip.generationSelection?.h3Reference?[keyPath:key] != nil},set:{ enabled in
        edit { s in var ref=s.h3Reference ?? H3ReferenceSettings();ref[keyPath:key]=enabled ? 1 : nil
          s.h3Reference = ref.visualConditionStrength == nil && ref.audioConditionStrength == nil ? nil : ref }
      }))
      if let value=clip.generationSelection?.h3Reference?[keyPath:key] {
        HStack {
          Slider(value:Binding(get:{value},set:{v in edit {$0.h3Reference?[keyPath:key]=v}}),in:0...1)
          Text(value,format:.number.precision(.fractionLength(2)))
        }
      }
    }
  }
}

struct H3LoRAInspector:View {
  @EnvironmentObject var store:StudioStore
  var attachment:Attachment
  private var settings:H3LoRASettings { attachment.h3LoRA ?? H3LoRASettings() }
  private func edit(_ body:@escaping(inout H3LoRASettings)->Void) {
    store.editClip { c in guard let i=c.attachments.firstIndex(where:{$0.id==attachment.id}) else {return}
      var value=c.attachments[i].h3LoRA ?? H3LoRASettings();body(&value);c.attachments[i].h3LoRA=value }
  }
  var body:some View {
    DisclosureGroup("H3 adapter settings") {
      Picker("Profile",selection:Binding(get:{settings.profile},set:{v in edit {$0.profile=v}})) {
        ForEach(H3LoRAProfile.allCases,id: \.self) { Text($0.rawValue).tag($0) }
      }
      Picker("QKV layout",selection:Binding(get:{settings.qkvLayout},set:{v in edit {$0.qkvLayout=v}})) {
        ForEach(H3LoRAQKVLayout.allCases,id: \.self) { Text($0.rawValue).tag($0) }
      }
      Stepper("Start after \(settings.startAfterEvaluations) evaluations",value:Binding(get:{settings.startAfterEvaluations},set:{v in edit {$0.startAfterEvaluations=v}}),in:0...99)
      Text("Turbo requires four Euler evaluations and immediate activation.").font(.caption2).foregroundStyle(.secondary)
      if attachment.h3LoRA != nil {
        Button("Use adapter defaults") { store.editClip { c in
          guard let i=c.attachments.firstIndex(where:{$0.id==attachment.id}) else {return};c.attachments[i].h3LoRA=nil
        }}
      }
    }
  }
}
