import AppKit
import StudioCore
import SwiftUI

struct H3LearnedUpscalerInspector:View {
  @EnvironmentObject var store:StudioStore
  var clip:Clip
  var selected:H3JointRefinementSettings
  @State private var enabledExpansionForLearned=false
  private func edit(_ body:@escaping(inout H3JointRefinementSettings)->Void) {
    store.editClip {c in guard var value=c.generationSelection?.h3Joint?.refinement else {return}
      body(&value);c.generationSelection?.h3Joint?.refinement=value}
  }
  var body:some View {
    VStack(alignment:.leading) {
      Toggle("Expanded spatial target · v2",isOn:Binding(get:{selected.expandedSpatialTarget==true},set:{value in
        edit {$0.expandedSpatialTarget=value ? true : nil}
        enabledExpansionForLearned=false
      })).disabled(selected.learnedUpscalerPath != nil)
      if let path=selected.learnedUpscalerPath {
        Text("Learned H3 upscaler: \(URL(fileURLWithPath:path).lastPathComponent)").font(.caption).textSelection(.enabled)
        Button("Use interpolation instead") {
          let needsExpandedCanvas=clip.generationWidth*clip.generationHeight>1376*768
          edit {value in
            value.learnedUpscalerPath=nil;value.learnedUpscalerHeaderSHA256=nil;value.resizeMethod = .bilinear
            if enabledExpansionForLearned && !needsExpandedCanvas {value.expandedSpatialTarget=nil}
          };enabledExpansionForLearned=false
        }
      }
      Button("Choose learned H3 latent upscaler…") {
        let panel=NSOpenPanel();panel.canChooseDirectories=false;panel.canChooseFiles=true;panel.allowsMultipleSelection=false
        guard panel.runModal() == .OK,let url=panel.url else {return}
        do {
          let model=try NativeH3LearnedUpscalerMetadata.inspect(path:url.path)
          enabledExpansionForLearned=enabledExpansionForLearned || selected.expandedSpatialTarget != true
          edit {value in
            value.learnedUpscalerPath=model.path;value.learnedUpscalerHeaderSHA256=model.headerSHA256
            value.expandedSpatialTarget=true;value.resizeMethod=nil
          }
        } catch {store.error=error.localizedDescription}
      }
      if selected.expandedSpatialTarget==true {
        Text("32-pixel grid; longest edge ≤1920, shortest edge ≤1088; area ≤1920×1088. Complete text, conditions and AV rows must fit the native 64000-row budget. Duration and conditioning can make a target unavailable.").font(.caption2).foregroundStyle(.secondary)
      }
      Text("The learned model's scale follows the target and source area. It has no separate adapter strength. Installed compatible weights are reused in place.").font(.caption2).foregroundStyle(.secondary)
    }
  }
}
