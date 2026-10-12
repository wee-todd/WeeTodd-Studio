import AppKit
import StudioCore
import SwiftUI

struct AudioPresentationOverlay: View {
  @EnvironmentObject var store: StudioStore
  let time: Double
  var body: some View {
    GeometryReader { geo in
      let aspect = Double(store.project.settings.width)/Double(max(1,store.project.settings.height))
      let width = min(geo.size.width,geo.size.height*aspect), height = min(geo.size.height,geo.size.width/aspect)
      if let image = AudioPresentationDrawing.image(size: CGSize(width:max(2,width),height:max(2,height)),
          project:store.project,features:store.audioVisualFeatures,time:time) {
        Image(decorative:image,scale:1).resizable().frame(width:width,height:height).frame(width:geo.size.width,height:geo.size.height)
      }
    }.allowsHitTesting(false).accessibilityLabel("Audio visualizer and captions")
  }
}
struct AudioPresentationControls: View {
  @EnvironmentObject var store: StudioStore
  private var value: AudioPresentation { store.project.audioPresentation ?? AudioPresentation() }
  func field<T>(_ key: WritableKeyPath<AudioPresentation,T>) -> Binding<T> {
    Binding(get:{ value[keyPath:key] },set:{ next in store.change {
      if $0.audioPresentation == nil { $0.audioPresentation = AudioPresentation() }; $0.audioPresentation?[keyPath:key] = next
    } })
  }
  func color(_ key: WritableKeyPath<AudioPresentation,String>) -> Binding<Color> {
    Binding(get:{ Color(nsColor:AudioPresentationDrawing.color(value[keyPath:key])) },set:{ next in
      guard let rgb = NSColor(next).usingColorSpace(.sRGB) else { return }
      field(key).wrappedValue = String(format:"%02X%02X%02X",Int(rgb.redComponent*255),Int(rgb.greenComponent*255),Int(rgb.blueComponent*255))
    })
  }
  var body: some View {
    ScrollView {
      VStack(alignment:.leading,spacing:12) {
        Text("Audio visuals & captions").font(.headline)
        Toggle("Visualizer in empty video sections",isOn:field(\.enabled))
        Picker("Style",selection:field(\.style)) { Text("Waveform").tag("waveform"); Text("Spectrum bars").tag("spectrum") }
        Picker("Reacts to",selection:field(\.source)) {
          Text("Audible mix").tag("mix"); Text("Voice").tag("voice"); Text("Music").tag("music"); Text("Voice + music").tag("voiceAndMusic")
        }
        HStack { ColorPicker("Color",selection:color(\.color),supportsOpacity:false); ColorPicker("Background",selection:color(\.background),supportsOpacity:false) }
        HStack { Text("Size"); Slider(value:field(\.size),in:0.1...0.8).accessibilityLabel("Visualizer size") }
        HStack { Text("Vertical position"); Slider(value:field(\.position),in:0.2...0.8).accessibilityLabel("Visualizer vertical position") }
        HStack { Text("Sensitivity"); Slider(value:field(\.sensitivity),in:0.1...4).accessibilityLabel("Visualizer sensitivity") }
        HStack { Text("Motion smoothing"); Slider(value:field(\.smoothing),in:0...1).accessibilityLabel("Visualizer smoothing") }
        Divider()
        Toggle("Voice captions",isOn:field(\.voiceCaptions))
        Toggle("Song lyrics",isOn:field(\.lyrics))
        Toggle("Show voice and lyrics together",isOn:field(\.showBothCaptions))
        HStack { Text("Caption size"); Slider(value:field(\.captionSize),in:16...96).accessibilityLabel("Caption font size") }
        ColorPicker("Caption color",selection:color(\.captionColor),supportsOpacity:false)
        Picker("Caption position",selection:field(\.captionPosition)) { Text("Bottom").tag("bottom");Text("Center").tag("center");Text("Top").tag("top") }
        Text("These settings apply to both preview and export. Select an audio region to generate or edit its captions.").font(.caption).foregroundStyle(.secondary)
        Divider()
        Text("Local caption model · English").font(.headline)
        Text(store.runtime.captionModelDirectory == nil ? "Not configured" : "Installed model selected").font(.caption)
        HStack {
          Button("Choose installed model…") { store.chooseCaptionModel(download:false) }
          Button("Download model…") { store.chooseCaptionModel(download:true) }.disabled(store.operationBusy)
        }
        Text("Native MLX recognition and timing. Singing can require corrections; review generated lyrics before export.").font(.caption).foregroundStyle(.secondary)
      }.padding(18)
    }.frame(width:440,height:650)
  }
}
struct AudioCaptionEditor: View {
  @EnvironmentObject var store: StudioStore
  let regionID: UUID
  @State private var text = ""
  var region: AudioRegion? { store.project.audio.first { $0.id == regionID } }
  func cueField<T>(_ cue: AudioCaption,_ key: WritableKeyPath<AudioCaption,T>) -> Binding<T> {
    Binding(get:{ region?.captions?.first(where:{$0.id == cue.id})?[keyPath:key] ?? cue[keyPath:key] },set:{ next in
      store.change { p in
        guard let i = p.audio.firstIndex(where:{$0.id == regionID}), let j = p.audio[i].captions?.firstIndex(where:{$0.id == cue.id}) else { return }
        p.audio[i].captions?[j][keyPath:key] = next
      }
    })
  }
  var body: some View {
    VStack(alignment:.leading,spacing:8) {
      Text("Captions / lyrics").font(.headline)
      if store.project.audioTracks.first(where: { $0.id == region?.trackID })?.role == .music {
        Toggle("Instrumental — no lyric captions",isOn:Binding(get:{ region?.instrumental ?? false },set:{ next in
          store.change { p in if let i=p.audio.firstIndex(where:{$0.id==regionID}) { p.audio[i].instrumental=next } }
        }))
      }
      Text("Optional script or lyrics — timing is measured from the recording.").font(.caption).foregroundStyle(.secondary)
      TextEditor(text:$text).frame(height:70).border(Color.secondary.opacity(0.3))
      HStack {
        Button("Generate captions") { Task { await store.generateCaptions(regionID:regionID,text:text) } }.disabled(store.operationBusy || store.runtime.captionModelDirectory == nil || region?.instrumental == true)
        Button("Settings…") { store.showAudioPresentation = true }
      }
      if store.operationBusy { Text(store.bridge.message).font(.caption); Button("Cancel") { store.bridge.cancel() } }
      Text("Cue times are seconds in the original audio. Trimming and moving the region keep captions aligned.").font(.caption).foregroundStyle(.secondary)
      ForEach(region?.captions ?? []) { cue in
        VStack(alignment:.leading,spacing:4) {
          HStack {
            Text("In"); TextField("Start",value:cueField(cue,\.start),format:.number)
            Text("Out"); TextField("End",value:cueField(cue,\.end),format:.number)
            Button { store.change { p in if let i=p.audio.firstIndex(where:{$0.id==regionID}) { p.audio[i].captions?.removeAll { $0.id==cue.id } } } } label: { Image(systemName:"trash") }
          }
          TextField("Caption",text:cueField(cue,\.text),axis:.vertical)
        }
      }
      Button("Add caption manually") {
        guard let r = region else { return }
        let start = max(r.sourceIn,min(r.sourceIn+r.duration-0.01,store.playhead-r.start+r.sourceIn))
        store.change { p in if let i=p.audio.firstIndex(where:{$0.id==regionID}) {
          if p.audio[i].captions == nil { p.audio[i].captions=[] }
          p.audio[i].captions?.append(AudioCaption(start:start,end:min(r.sourceIn+r.duration,start+3),text:"Caption"))
        } }
      }
    }.onAppear {
      guard let r = region, let a=store.project.assets.first(where:{$0.id==r.assetID}) else { return }
      text = a.voiceGeneration?.draft.text ?? a.musicGeneration?.draft.lyrics ?? ""
    }
  }
}
