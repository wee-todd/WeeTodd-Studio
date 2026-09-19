import Foundation
import StudioCore

extension StudioStore {
  func useReference(_ source: MediaAsset, action: ReferenceAction) async {
    guard !operationBusy, let clip = selectedClip else { return }
    let session = documentSessionID
    let projectID = project.id
    let key = generationRequestKey(for: clip)
    var asset = source
    do {
      if let method = action.preparation {
        notice = "Preparing \(action.label.lowercased())…"
        let result = try await bridge.invoke("prepare-reference", runtime: runtime, payload: [
          "path": source.path, "method": method, "duration": clip.duration,
          "width": clip.generationWidth, "height": clip.generationHeight, "fps": 24
        ], output: dataDirectory.appendingPathComponent("References"))
        guard let path = result["path"] as? String, let rawKind = result["kind"] as? String,
          let kind = AssetKind(rawValue: rawKind) else {
          throw StudioError.invalid("Reference preparation returned no media.")
        }
        asset = MediaAsset(name: source.name + (method == "sheet" ? " · reference sheet" : " · edge guide"),
          kind: kind, path: path, scope: .clip, owner: clip.id)
        asset.width = result["width"] as? Int ?? 0
        asset.height = result["height"] as? Int ?? 0
        asset.duration = result["duration"] as? Double ?? 0
        asset.fps = result["fps"] as? Double ?? 0
      }
      guard session == documentSessionID, projectID == project.id else {
        return // Prepared output remains on disk; never attach to a replacement document.
      }
      guard let index = project.clips.firstIndex(where: { $0.id == clip.id }),
        selectedClipID == clip.id, generationRequestKey(for: project.clips[index]) == key,
        allAssets.contains(source) else {
        if action.preparation != nil {
          asset.scope = .project; asset.owner = nil
          change { $0.assets.append(asset) }
        }
        notice = "The shot changed. Prepared media is saved in Project Assets; attach it when ready."
        return
      }
      var updated = project.clips[index]
      try updated.attachReference(asset, action: action)
      change {
        if action.preparation != nil { $0.assets.append(asset) }
        $0.clips[index] = updated
      }
      selectedAssetID = asset.id
      notice = action.detail
    } catch { self.error = error.localizedDescription }
  }
}
