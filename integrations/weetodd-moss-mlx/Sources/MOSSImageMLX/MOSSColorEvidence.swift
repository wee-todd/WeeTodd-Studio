import CoreGraphics
import Foundation
import ImageLayout

enum MOSSColorEvidence {
  struct Border { let caption, hex: String }
  private static func pixels(_ image: CGImage, bounds: NormalizedLayoutRect? = nil) -> [UInt8] {
    var input = image
    if let b = bounds {
      let r = CGRect(x:b.x*Double(image.width),y:b.y*Double(image.height),width:b.width*Double(image.width),height:b.height*Double(image.height)).integral.intersection(CGRect(x:0,y:0,width:image.width,height:image.height))
      guard let crop = image.cropping(to:r) else { return [] }; input = crop
    }
    var bytes = [UInt8](repeating:0,count:64*64*4)
    let ok = bytes.withUnsafeMutableBytes { data -> Bool in
      guard let c = CGContext(data:data.baseAddress,width:64,height:64,bitsPerComponent:8,bytesPerRow:256,space:CGColorSpace(name:CGColorSpace.sRGB)!,bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return false }
      c.interpolationQuality = .medium;c.draw(input,in:CGRect(x:0,y:0,width:64,height:64));return true
    }
    return ok ? bytes : []
  }
  static func grayscaleFraction(_ image: CGImage, bounds: NormalizedLayoutRect? = nil) -> Double {
    let bytes = pixels(image,bounds:bounds);var count=0,gray=0
    for p in stride(from:0,to:bytes.count,by:4) where bytes[p+3] > 250 {
      count += 1
      if Int(max(bytes[p],bytes[p+1],bytes[p+2]))-Int(min(bytes[p],bytes[p+1],bytes[p+2])) <= 8 { gray += 1 }
    }
    return count == 0 ? 0 : Double(gray)/Double(count)
  }
  static func uniformBorder(_ image: CGImage) -> Border? {
    let bytes=pixels(image);guard !bytes.isEmpty else{return nil}
    let samples=(0..<4096).filter { $0%64<2 || $0%64>=62 || $0/64<2 || $0/64>=62 }.filter { bytes[$0*4+3]>250 }
    guard samples.count>400 else{return nil}
    let rgb=(0..<3).map { channel in samples.map{Int(bytes[$0*4+channel])}.sorted()[samples.count/2] }
    let matching=samples.filter { p in (0..<3).allSatisfy { abs(Int(bytes[p*4+$0])-rgb[$0])<=6 } }.count
    guard Double(matching)/Double(samples.count)>=0.95 else{return nil}
    let r=Double(rgb[0])/255,g=Double(rgb[1])/255,b=Double(rgb[2])/255
    let high=max(r,g,b),low=min(r,g,b),delta=high-low,saturation=high==0 ? 0 : delta/high
    var hue=0.0
    if delta>0 {
      if high==r { hue=(g-b)/delta } else if high==g { hue=2+(b-r)/delta } else { hue=4+(r-g)/delta }
      hue/=6;if hue<0{hue+=1}
    }
    let name:String
    if saturation<0.1 { name=high<0.12 ? "black" : high>0.9 ? "white" : high<0.4 ? "dark gray" : "gray" }
    else {
      let color=hue<0.04 ? "red" : hue<0.1 ? "orange" : hue<0.17 ? "yellow" : hue<0.27 ? "olive-green" : hue<0.46 ? "green" : hue<0.56 ? "teal" : hue<0.72 ? "blue" : hue<0.84 ? "purple" : hue<0.94 ? "pink" : "red"
      name=(high<0.3 ? "dark " : saturation<0.4 ? "muted " : "")+color
    }
    let hex=String(format:"#%02X%02X%02X",rgb[0],rgb[1],rgb[2])
    return .init(caption:"A flat, uniform \(name) canvas background (\(hex)).",hex:hex)
  }
  static func grayscaleCaption(_ caption: String) -> String {
    let corrected=caption.replacingOccurrences(of:#"\b(color|colour|full-color|full-colour) photograph\b"#,with:"grayscale photograph",options:[.regularExpression,.caseInsensitive])
    return "Black-and-white grayscale depiction. " + corrected
  }
}
