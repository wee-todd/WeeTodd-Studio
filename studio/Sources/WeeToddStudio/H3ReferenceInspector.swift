import AppKit
import StudioCore
import SwiftUI

struct H3ReferenceInspector:View {
  @EnvironmentObject var store:StudioStore
  var attachment:Attachment
  var isVideo:Bool
  var isImage:Bool = true
  var showsPlacement:Bool = true
  private func edit(_ body:@escaping(inout H3ReferencePlacement)->Void) {
    store.editClip { c in
      guard let i=c.attachments.firstIndex(where:{$0.id==attachment.id}) else {return}
      var value=c.attachments[i].h3ReferencePlacement ?? H3ReferencePlacement();body(&value)
      c.attachments[i].h3ReferencePlacement = value.isEmpty ? nil : value
    }
  }
  private var frame:Int {
    if case .index(let i)=attachment.h3ReferencePlacement?.frame {return i};return 0
  }
  var body:some View {
    DisclosureGroup("H3 reference placement") {
      if showsPlacement {
      Picker("Frame",selection:Binding(get:{
        switch attachment.h3ReferencePlacement?.frame {
        case nil:return "default"
        case .last:return "last"
        case .index:return "index"
        }
      },set:{choice in edit {$0.frame=choice == "last" ? .last : choice == "index" ? .index(0) : nil}})) {
        Text("Recipe default").tag("default");Text("Specific frame").tag("index");Text("Last visible frame").tag("last")
      }
      if case .index=attachment.h3ReferencePlacement?.frame {
        Stepper("Frame \(frame) · \(Double(frame)/24,specifier:"%.3f") s",value:Binding(get:{frame},set:{i in edit {$0.frame = .index(i)}}),in:0...max(0,Int(ceil((store.selectedClip?.duration ?? 0)*24))-1))
      }
      }
      if isImage && !isVideo {
        Toggle("Override image pixel budget",isOn:Binding(get:{attachment.h3ReferencePlacement?.imagePixelBudgetPercent != nil},set:{ enabled in edit {$0.imagePixelBudgetPercent=enabled ? 100 : nil} }))
        if let percent=attachment.h3ReferencePlacement?.imagePixelBudgetPercent {
          Stepper("Image budget · \(percent)% of output pixels",value:Binding(get:{percent},set:{v in edit {$0.imagePixelBudgetPercent=v}}),in:50...400,step:25)
          Text("Downscales large references without enlarging small originals.").font(.caption2).foregroundStyle(.secondary)
        }
      }
      if isVideo {
        Toggle("Override movie preparation",isOn:Binding(get:{attachment.h3ReferencePlacement?.videoSizePolicy != nil || attachment.h3ReferencePlacement?.videoTemporalDensity != nil},set:{ enabled in edit {$0.videoSizePolicy=enabled ? .matchOutput : nil;$0.videoTemporalDensity=enabled ? .full : nil} }))
        if attachment.h3ReferencePlacement?.videoSizePolicy != nil || attachment.h3ReferencePlacement?.videoTemporalDensity != nil {
          Picker("Movie canvas",selection:Binding(get:{attachment.h3ReferencePlacement?.videoSizePolicy ?? .matchOutput},set:{v in edit {$0.videoSizePolicy=v}})) {
            Text("Match output budget").tag(H3VideoReferenceSizePolicy.matchOutput)
            Text("Native H3 budget").tag(H3VideoReferenceSizePolicy.nativeH3)
          }
          Picker("Temporal density",selection:Binding(get:{attachment.h3ReferencePlacement?.videoTemporalDensity ?? .full},set:{v in edit {$0.videoTemporalDensity=v}})) {
            ForEach(H3VideoReferenceTemporalDensity.allCases,id: \.self) { Text($0.rawValue.capitalized).tag($0) }
          }
          Text("Density reduces video encoding work while keeping the original reference duration and audio clock.").font(.caption2).foregroundStyle(.secondary)
        }
        if let sound=attachment.h3ReferencePlacement?.soundtrackPath {
          Text(URL(fileURLWithPath:sound).lastPathComponent).font(.caption).textSelection(.enabled)
          Button("Use embedded soundtrack") { edit {$0.soundtrackPath=nil} }
        }
        Button("Choose replacement soundtrack…") {
          let panel=NSOpenPanel();panel.canChooseFiles=true;panel.canChooseDirectories=false;panel.allowsMultipleSelection=false
          if panel.runModal() == .OK,let url=panel.url {edit {$0.soundtrackPath=url.path}}
        }
        Text("The movie and its replacement soundtrack remain one synchronized reference.").font(.caption2).foregroundStyle(.secondary)
      }
    }
  }
}
