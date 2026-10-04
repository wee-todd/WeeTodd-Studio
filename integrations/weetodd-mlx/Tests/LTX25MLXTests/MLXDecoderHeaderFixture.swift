import Foundation
import XCTest

/// Sparse descriptor-only fixture. Admission reads the real regular header;
/// no production dummy-path exception, tensor payload read or model execution.
enum MLXDecoderHeaderFixture {
  static func convolutional(test:XCTestCase) throws -> URL {
    let url=FileManager.default.temporaryDirectory.appendingPathComponent("conv-header-\(UUID().uuidString).safetensors")
    test.addTeardownBlock { try? FileManager.default.removeItem(at:url) }
    var shapes:[String:[Int]]=[:]
    func convolution(_ name:String,_ input:Int,_ output:Int) {
      shapes[name+".weight"]=[output,input,3,3,3];shapes[name+".bias"]=[output]
    }
    convolution("decoder.conv_in.conv",128,1024)
    let channels=[1024,512,512,256,128],counts=[2,2,4,6,4],outputs=[4096,4096,512,512]
    for stage in 0..<5 {
      for block in 0..<counts[stage] { for conv in 1...2 {
        convolution("decoder.up_blocks.\(stage*2).res_blocks.\(block).conv\(conv).conv",channels[stage],channels[stage])
      } }
      if stage<4 { convolution("decoder.up_blocks.\(stage*2+1).conv.conv",channels[stage],outputs[stage]) }
    }
    convolution("decoder.conv_out.conv",128,48)
    shapes["per_channel_statistics.mean-of-means"]=[128];shapes["per_channel_statistics.std-of-means"]=[128]
    let config=try JSONSerialization.data(withJSONObject:["vae":["_class_name":"CausalVideoAutoencoder","spatial_padding_mode":"zeros","timestep_conditioning":false,"causal_decoder":false]])
    var header:[String:Any]=["__metadata__":["model_version":"2.5.0","config":String(data:config,encoding:.utf8)!]],offset=0
    for name in shapes.keys.sorted() {
      let shape=shapes[name]!,bytes=shape.reduce(1,*)*2
      header[name]=["dtype":"BF16","shape":shape,"data_offsets":[offset,offset+bytes]];offset+=bytes
    }
    let raw=try JSONSerialization.data(withJSONObject:header,options:.sortedKeys);var length=UInt64(raw.count).littleEndian
    var data=withUnsafeBytes(of:&length) { Data($0) };data.append(raw);try data.write(to:url,options:.withoutOverwriting)
    let file=try FileHandle(forWritingTo:url);try file.truncate(atOffset:UInt64(data.count+offset));try file.close()
    return url
  }
}
extension XCTestCase {
  func decoderRequestBase(_ fields:[String:Any]) throws -> [String:Any] {
    var value=fields;value["video_checkpoint"]=try MLXDecoderHeaderFixture.convolutional(test:self).path;return value
  }
}
