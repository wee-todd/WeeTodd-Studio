import Foundation

public struct MOSSRegionalPrompt: Codable, Equatable, Sendable {
  public enum Role: String, Codable, Sendable { case background, subject, detail, object, lettering }
  public let role: Role
  public let bounds: NormalizedLayoutRect?
  public let prompt: String
  public let regionID: UUID?
  public let masks: [NormalizedLayoutRect]?
  public init(role: Role, bounds: NormalizedLayoutRect?, prompt: String, regionID: UUID? = nil, masks: [NormalizedLayoutRect]? = nil) {
    self.role = role; self.bounds = bounds; self.prompt = prompt; self.regionID = regionID; self.masks = masks
  }
}

public enum MOSSRegionalLayout {
  public static func groups(_ draft: ImageLayoutDraft) throws -> [MOSSRegionalPrompt] {
    _ = try MOSSLayoutCompiler.compile(draft)
    let subjects = draft.regions.filter { $0.type == .character && $0.parentID == nil }
    guard subjects.count <= 8, draft.regions.filter({ $0.type == .text }).allSatisfy({ $0.parentID == nil }) else {
      throw ImageLayoutError.invalid("Regional layout supports up to eight whole subjects and independent text boxes.")
    }
    let families = subjects.map { ImageLayoutEditing.descendants(of: $0.id, in: draft) }
    let owned = families.reduce(into: Set<UUID>()) { $0.formUnion($1) }
    let style = styleOnly(draft.summary.caption)
    func prompt(_ regions: [ImageLayoutRegion], purpose: String, foreground: Bool, medium: String? = nil) throws -> String {
      var group = draft; group.regions = regions
      var treatment = style
      if foreground {
        treatment = treatment.replacingOccurrences(of:"Monochrome photographic elements within colored graphic design.",with:"Graphic composition.")
        treatment = treatment.replacingOccurrences(of:"Textured paper design.",with:"")
        group.summary.palette = nil
        let own = medium?.lowercased() ?? ""
        let explicit = own.hasPrefix("black-and-white grayscale depiction.") || own.range(of:#"\b(monochrome|grayscale|black-and-white|black and white)\s+(photo(?:graph)?|photography|portrait|illustration|drawing|painting)\b"#,options:.regularExpression) != nil
        if explicit {
          treatment = ["illustrat","draw","paint","cartoon"].contains(where:own.contains) ? "Black-and-white grayscale illustration." : own.contains("photo") ? "Black-and-white grayscale photography." : "Black-and-white grayscale depiction."
          group.summary.meta = ["grayscale","monochrome"]
          group.summary.palette = ["#000000","#808080","#FFFFFF"]
        }
      }
      group.summary.caption = treatment + " " + purpose
      group.guideImagePath = nil; group.guideOpacity = nil; group.sourceReconstruction = nil
      return try MOSSLayoutCompiler.compile(group).effectivePrompt
    }
    let canvas = draft.regions.filter { $0.analysisRole == .canvasBackground }
    var result: [MOSSRegionalPrompt] = [.init(role:.background,bounds:nil,
      prompt:try prompt(canvas,purpose:"Background of the composition.",foreground:false))]
    for (index, subject) in subjects.enumerated() {
      let family = draft.regions.filter { families[index].contains($0.id) }
      let faces = family.filter { $0.analysisRole == .face }
      let faceIDs = faces.reduce(into:Set<UUID>()) { $0.formUnion(ImageLayoutEditing.descendants(of:$1.id,in:draft)) }
      var masks = family.map(\.bounds)
      if let face = faces.first {
        let b=subject.bounds,f=face.bounds
        if f.y+f.height/2 < b.y+b.height*0.45 && b.height>f.height*1.5 {
          let top=max(b.y,min(b.y+b.height,f.y+f.height-b.height*0.06))
          let torso=NormalizedLayoutRect(x:b.x,y:top,width:b.width,height:b.y+b.height-top)
          let x=max(0,f.x-f.width*0.35),y=max(0,f.y-f.height*0.35)
          let head=NormalizedLayoutRect(x:x,y:y,width:min(1,f.x+f.width*1.35)-x,height:min(1,f.y+f.height*1.35)-y)
          masks = [torso,head] + family.filter { $0.id != subject.id }.map(\.bounds)
        }
      }
      result.append(.init(role:.subject,bounds:subject.bounds,
        prompt:try prompt(family.filter { !faceIDs.contains($0.id) },purpose:"One complete subject and its own body details.",foreground:true,medium:subject.caption),regionID:subject.id,masks:masks))
      for face in faces {
        var detail = draft.regions.filter { ImageLayoutEditing.descendants(of:face.id,in:draft).contains($0.id) }
        for i in detail.indices where detail[i].id == face.id { detail[i].parentID = nil }
        result.append(.init(role:.detail,bounds:face.bounds,prompt:try prompt(detail,purpose:"Facial detail of this same existing person.",foreground:true,medium:face.caption),regionID:face.id,masks:detail.map(\.bounds)))
      }
    }
    let roots = draft.regions.filter { $0.parentID == nil && $0.type == .object && $0.analysisRole != .canvasBackground && !owned.contains($0.id) }
    for item in roots {
      let family=ImageLayoutEditing.descendants(of:item.id,in:draft)
      let regions=draft.regions.filter { family.contains($0.id) }
      result.append(.init(role:.object,bounds:item.bounds,prompt:try prompt(regions,purpose:"An element of the same composition.",foreground:true,medium:item.caption),regionID:item.id,masks:regions.map(\.bounds)))
    }
    let text = draft.regions.filter { $0.type == .text }
    if !text.isEmpty { result.append(.init(role:.lettering,bounds:nil,prompt:try prompt(text,purpose:"Typography and graphic lettering of the composition.",foreground:true),masks:text.map(\.bounds))) }
    guard result.count<=62 else { throw ImageLayoutError.invalid("Too many independent regional streams.") }
    return result
  }
  static func styleOnly(_ caption: String) -> String {
    let lower = caption.lowercased()
    func has(_ pattern: String) -> Bool { lower.range(of:pattern,options:.regularExpression) != nil }
    var evidence:[String]=[]
    if lower.contains("monochrome") { evidence.append(["cover","poster","collage","magazine"].contains(where:lower.contains) ? "Monochrome photographic elements within colored graphic design." : "Monochrome photography.") }
    else if ["photograph","cinematic","film still"].contains(where:lower.contains) { evidence.append("Photographic visual treatment.") }
    else if ["illustrat","painted","painting"].contains(where:lower.contains) { evidence.append("Illustrated visual treatment.") }
    if has(#"\bwarm\b"#) && has(#"\bcinematic\b"#) { evidence.append("Warm cinematic lighting.") }
    else if has(#"\bwarm\b"#) && has(#"\blight(ing)?\b"#) { evidence.append("Warm lighting.") }
    let styles:[(String,String)] = [(#"\bsoft\b.*\blight(ing)?\b"#,"Soft lighting."),(#"\beven\b.*\blight(ing)?\b"#,"Even lighting."),(#"\bhigh[- ]contrast\b"#,"High-contrast visual treatment."),(#"\bdramatic\b.*\blight(ing)?\b"#,"Dramatic lighting."),(#"\b(textured|weathered|aged)\b.*\bpaper\b|\bpaper\b.*\btexture"#,"Textured paper design."),(#"\bfilm grain\b"#,"Visible film grain."),(#"\bwide[- ]angle\b"#,"Wide-angle camera view."),(#"\blow[- ]angle\b"#,"Low-angle camera view."),(#"\bwatercolor\b"#,"Watercolor medium."),(#"\boil paint(ing)?\b"#,"Oil painting medium."),(#"\bclaymation\b"#,"Claymation medium.")]
    for (pattern,description) in styles where has(pattern) { evidence.append(description) }
    let colors:[(String,String)] = [("navy(?:[- ]blue)?","deep navy-blue"),("olive(?:[- ]green)?","muted olive-green"),("dark[- ]blue","dark blue"),("off[- ]white","off-white"),("cream","cream"),("black","black"),("white","white"),("red","red"),("green","green"),("blue","blue")]
    for (pattern,name) in colors {
      if has(#"\bbackground(?:\s+is)?\s+(?:(?:a|an|the|deep|dark|muted|plain|flat)\s+){0,4}"#+pattern+#"\b|\b"#+pattern+#"\s+background\b"#) { evidence.append("A "+name+" background.");break }
    }
    let result=evidence.joined(separator:" ");return result.isEmpty ? "A cohesive visual composition." : result
  }
}
