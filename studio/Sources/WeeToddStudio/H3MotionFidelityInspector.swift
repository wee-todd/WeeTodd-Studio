import AppKit
import StudioCore
import SwiftUI

struct H3MotionFidelityInspector:View {
  @EnvironmentObject var store:StudioStore
  var clip:Clip
  private var settings:H3MotionFidelitySettings? {clip.generationSelection?.h3MotionFidelity}
  private func edit(_ body:@escaping(inout H3MotionFidelitySettings)->Void) {
    store.editClip { c in
      var selection=c.generationSelection ?? GenerationSelection(task:c.inferredTask,preset:.custom)
      var value=selection.h3MotionFidelity ?? H3MotionFidelitySettings(sourceVideo:"",experimentalEnabled:true)
      body(&value);selection.h3MotionFidelity=value;c.generationSelection=selection
    }
  }
  private func binding<T>(_ key:WritableKeyPath<H3MotionFidelitySettings,T>,default fallback:T)->Binding<T> {
    Binding(get:{settings?[keyPath:key] ?? fallback},set:{value in edit {$0[keyPath:key]=value}})
  }
  var body:some View {
    DisclosureGroup("Motion repair · experimental") {
      Toggle("Enable Motion Fidelity",isOn:Binding(get:{settings != nil},set:{enabled in
        store.editClip { c in
          var selection=c.generationSelection ?? GenerationSelection(task:c.inferredTask,preset:.custom)
          selection.h3MotionFidelity=enabled ? H3MotionFidelitySettings(sourceVideo:"",experimentalEnabled:true) : nil
          c.generationSelection=selection
        }
      }))
      if let settings {
        Text(settings.sourceVideo.isEmpty ? "Choose the source movie" : URL(fileURLWithPath:settings.sourceVideo).lastPathComponent).font(.caption).textSelection(.enabled)
        Button("Choose source movie…") {
          let panel=NSOpenPanel();panel.canChooseFiles=true;panel.canChooseDirectories=false;panel.allowsMultipleSelection=false
          if panel.runModal() == .OK,let url=panel.url {edit {$0.sourceVideo=url.path}}
        }
        if !clip.sourcePath.isEmpty {
          Button("Use current accepted take") {edit {$0.sourceVideo=clip.sourcePath;$0.sourceIn=clip.sourceIn}}
        }
        TextField("Source in (seconds)",value:binding(\.sourceIn,default:0),format:.number)
        Picker("Repair",selection:binding(\.mode,default:.adaptive)) {
          Text("Adaptive").tag(H3MotionFidelitySettings.Mode.adaptive)
          Text("Uniform holds").tag(H3MotionFidelitySettings.Mode.uniform)
        }
        LabeledContent("Strength",value:settings.strength.formatted(.number.precision(.fractionLength(2))))
        Slider(value:binding(\.strength,default:0.5),in:0.01...1)
        Stepper("Maximum hold · \(settings.maxHold) frames",value:binding(\.maxHold,default:2),in:2...4)
        if settings.mode == .adaptive {
          LabeledContent("Sensitivity",value:settings.sensitivity.formatted(.number.precision(.fractionLength(2))))
          Slider(value:binding(\.sensitivity,default:0.5),in:0...1)
        }
        Stepper("Expanded frame limit · \(settings.maxFrames)",value:binding(\.maxFrames,default:345),in:73...345)
        Toggle("Override repair evaluations",isOn:Binding(get:{settings.evaluations != nil},set:{enabled in edit {$0.evaluations=enabled ? 8 : nil}}))
        if let count=settings.evaluations {
          Stepper("Repair evaluations · \(count)",value:Binding(get:{count},set:{n in edit {$0.evaluations=n}}),in:1...64)
        }
        Text("Uses the clip's duration, canvas and seed. Source must be 24 fps at the same canvas. Requires independent T2VA, at least 15 base evaluations, and standard immediate LoRAs. Original timing and audio are preserved.").font(.caption2).foregroundStyle(.secondary)
      }
    }
  }
}
