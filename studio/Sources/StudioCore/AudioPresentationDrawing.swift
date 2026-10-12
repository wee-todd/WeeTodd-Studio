import AppKit

/// Shared normalized drawing for live preview and exported frames.
public enum AudioPresentationDrawing {
  public static func color(_ hex: String) -> NSColor {
    let value = UInt32(hex, radix: 16) ?? 0xFFFFFF
    return NSColor(srgbRed: CGFloat((value >> 16) & 255)/255, green: CGFloat((value >> 8) & 255)/255,
                   blue: CGFloat(value & 255)/255, alpha: 1)
  }
  public static func draw(in cg: CGContext, size: CGSize, project: StudioProject,
                          features: AudioVisualFeatures?, time: Double) {
    guard let settings = project.audioPresentation else { return }
    cg.saveGState(); defer { cg.restoreGState() }
    cg.scaleBy(x: size.width/1920, y: size.height/1080)
    if settings.enabled && !project.hasVideo(at: time) {
      cg.setFillColor(color(settings.background).cgColor); cg.fill(CGRect(x: 0,y: 0,width: 1920,height: 1080))
      let values = features?.values(at: time, style: settings.style, smoothing: settings.smoothing) ?? []
      cg.setStrokeColor(color(settings.color).cgColor); cg.setFillColor(color(settings.color).cgColor)
      let center = 1080*(1-settings.position), amplitude = 1080*settings.size/2
      if settings.style == "spectrum" {
        let count = max(1,values.count), step = 1536/Double(count)
        for (i,value) in values.enumerated() {
          let h = max(2,min(1,max(0,value*settings.sensitivity))*amplitude*2)
          let rect = CGRect(x: 192+Double(i)*step+3, y: center-h/2, width: max(2,step-6), height: h)
          cg.addPath(CGPath(roundedRect: rect, cornerWidth: 4, cornerHeight: 4, transform: nil)); cg.fillPath()
        }
      } else {
        cg.setLineWidth(4); cg.setLineCap(.round); cg.setLineJoin(.round)
        cg.move(to: CGPoint(x: 192,y: center))
        for (i,value) in values.enumerated() {
          cg.addLine(to: CGPoint(x: 192+1536*Double(i)/Double(max(1,values.count-1)), y: center+max(-1,min(1,value*settings.sensitivity))*amplitude))
        }
        if values.isEmpty { cg.addLine(to: CGPoint(x: 1728,y: center)) }
        cg.strokePath()
      }
    }
    for title in project.titles where time >= title.start && time < title.start + title.duration && !title.text.isEmpty {
      NSGraphicsContext.saveGraphicsState()
      NSGraphicsContext.current = NSGraphicsContext(cgContext: cg, flipped: false)
      let paragraph = NSMutableParagraphStyle(); paragraph.alignment = .center
      let fontSize = Double(title.fontSize)*1920/Double(max(1,project.settings.width))
      let string = NSAttributedString(string:title.text, attributes:[.font:NSFont.systemFont(ofSize:fontSize,weight:.semibold),
        .foregroundColor:NSColor.white,.paragraphStyle:paragraph])
      let bounds = string.boundingRect(with:CGSize(width:1536,height:400),options:[.usesLineFragmentOrigin,.usesFontLeading])
      let h = ceil(bounds.height)+16, y = title.position == "center" ? (1080-h)/2 : 108
      cg.setFillColor(NSColor.black.withAlphaComponent(0.35).cgColor)
      cg.fill(CGRect(x:180,y:y-8,width:1560,height:h+16))
      string.draw(with:CGRect(x:192,y:y,width:1536,height:h),options:[.usesLineFragmentOrigin,.usesFontLeading])
      NSGraphicsContext.restoreGraphicsState()
    }
    let text = project.captionText(at: time)
    guard !text.isEmpty else { return }
    NSGraphicsContext.saveGraphicsState(); defer { NSGraphicsContext.restoreGraphicsState() }
    NSGraphicsContext.current = NSGraphicsContext(cgContext: cg, flipped: false)
    let paragraph = NSMutableParagraphStyle(); paragraph.alignment = .center
    let attributes: [NSAttributedString.Key: Any] = [.font: NSFont.systemFont(ofSize: settings.captionSize,weight: .semibold),
      .foregroundColor: color(settings.captionColor), .paragraphStyle: paragraph]
    let string = NSAttributedString(string: text, attributes: attributes)
    let bounds = string.boundingRect(with: CGSize(width: 1536,height: 400), options: [.usesLineFragmentOrigin,.usesFontLeading])
    let h = min(400,ceil(bounds.height)+16)
    let y = settings.captionPosition == "top" ? 1080-h-70 : settings.captionPosition == "center" ? (1080-h)/2 : 70
    let rect = CGRect(x: 180,y: y-8,width: 1560,height: h+16)
    cg.setFillColor(NSColor.black.withAlphaComponent(0.65).cgColor)
    cg.addPath(CGPath(roundedRect: rect,cornerWidth: 10,cornerHeight: 10,transform:nil)); cg.fillPath()
    string.draw(with: CGRect(x:192,y:y,width:1536,height:h), options:[.usesLineFragmentOrigin,.usesFontLeading])
  }
  public static func image(size: CGSize, project: StudioProject, features: AudioVisualFeatures?, time: Double) -> CGImage? {
    guard size.width > 0, size.height > 0, size.width <= 8192, size.height <= 8192,
      let cg = CGContext(data:nil,width:Int(size.width),height:Int(size.height),bitsPerComponent:8,bytesPerRow:0,
        space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    draw(in:cg,size:size,project:project,features:features,time:time)
    return cg.makeImage()
  }
}
