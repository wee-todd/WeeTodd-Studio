import AVFoundation
import AppKit
import StudioCore

/// Emits raw BGRA frames to a pipe consumed by the media worker's encoder.
enum AudioPresentationExport {
  static func run(arguments: [String]) throws {
    guard arguments.count == 5, let width = Int(arguments[2]), let height = Int(arguments[3]),
      let fps = Double(arguments[4]), width > 0, height > 0, width <= 8192, height <= 8192,
      fps.isFinite, fps >= 1, fps <= 240 else { throw StudioError.invalid("Invalid presentation render geometry") }
    let project = try JSONDecoder().decode(StudioProject.self, from: Data(contentsOf: URL(fileURLWithPath:arguments[0])))
    try project.audioPresentation?.validate(); try project.validateAudio()
    let features = arguments[1] == "-" ? nil : try JSONDecoder().decode(AudioVisualFeatures.self,from:Data(contentsOf:URL(fileURLWithPath:arguments[1])))
    let duration = project.playbackDuration
    guard duration.isFinite, duration > 0, duration <= 3600 else { throw StudioError.invalid("Audio presentation supports up to one hour") }
    let bytes = width*height*4
    let pixels = UnsafeMutableRawPointer.allocate(byteCount: bytes,alignment: 64); defer { pixels.deallocate() }
    guard let cg = CGContext(data:pixels,width:width,height:height,bitsPerComponent:8,bytesPerRow:width*4,
      space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue) else { throw StudioError.invalid("Cannot allocate presentation frame") }
    for frame in 0..<Int(ceil(duration*fps)) {
      try autoreleasepool {
        cg.clear(CGRect(x:0,y:0,width:width,height:height))
        AudioPresentationDrawing.draw(in:cg,size:CGSize(width:width,height:height),project:project,features:features,time:Double(frame)/fps)
        // Quartz bitmap bytes and CGImage use the same row order for preview/export.
        try FileHandle.standardOutput.write(contentsOf:Data(bytesNoCopy:pixels,count:bytes,deallocator:.none))
      }
    }
  }
}
