import Foundation
import NNC
import Metal
import Darwin
extension H3WorkerMain {
 static func runStack(fixture:String,checkpoint:String,adapter:String,output:String,iterations:Int,serving:Bool=false) throws {
  let env=ProcessInfo.processInfo.environment
  guard ["copied","bounded"].contains(env["WEETODD_NNC_BUFFER_IO"] ?? "copied") else { throw ProbeError.invalid("Unknown buffer IO policy") }
  guard ["eager","block"].contains(env["WEETODD_NNC_RESIDENCY"] ?? "eager") else { throw ProbeError.invalid("Unknown block residency") }
  let start=Int(env["WEETODD_NNC_BLOCK_START"] ?? "0") ?? -1
  let count=Int(env["WEETODD_NNC_BLOCK_COUNT"] ?? "50") ?? -1
  guard (0..<50).contains(start),(1...50).contains(count),start+count<=50 else { throw ProbeError.invalid("Invalid block range") }
  guard (1...10).contains(iterations) else { throw ProbeError.invalid("Invalid iteration count") }
  let folder=URL(fileURLWithPath:output);try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
  let reportURL=folder.appendingPathComponent("nnc.json")
  if FileManager.default.fileExists(atPath:reportURL.path) { try FileManager.default.removeItem(at:reportURL) }
  let data=try SafeTensorReader(fixture),weights=try SafeTensorReader(checkpoint),lora=try SafeTensorReader(adapter)
  guard let shape=data.records["x"]?.shape,shape.count==2,shape[1]==5376,(1...40000).contains(shape[0]) else { throw ProbeError.invalid("Invalid stack input") }
  let rows=shape[0],half=env["WEETODD_NNC_PRECISION"]=="fp16"
  let graph=DynamicGraph(),stream=StreamContext(.GPU(0))
  var inputs:[DynamicGraph.AnyTensor]=[]
  func tensor(_ source:FloatTensor,integer:Bool=false,residual:Bool=false) throws -> DynamicGraph.AnyTensor {
   guard source.values.allSatisfy({$0.isFinite}) else { throw ProbeError.invalid("Nonfinite fixture") }
   if integer { return graph.variable(Tensor<Int32>(source.values.map{Int32($0)},kind:.CPU,format:.NHWC,shape:TensorShape(source.shape)).toGPU(0)) }
   if half && residual { return graph.variable(hostTensor(source.values,shape:TensorShape(source.shape)).toGPU(0)) }
   if half {
    let values=source.values.map{Float16($0)}
    guard values.allSatisfy({$0.isFinite}) else { throw ProbeError.invalid("FP16 fixture overflow") }
    return graph.variable(hostTensor(values,shape:TensorShape(source.shape)).toGPU(0))
   }
   return graph.variable(makeBF16(source).toGPU(0))
  }
  let indices=try data.read("indices")
  guard indices.shape==[rows],let tableShape=data.records["block\(start).mod0"]?.shape,tableShape.count==2,tableShape[0]>0,tableShape[0]<=300,tableShape[1]==5376,indices.values.allSatisfy({$0.isFinite && $0>=0 && $0<Float(tableShape[0]) && $0.rounded()==$0}) else { throw ProbeError.invalid("Invalid modulation indices") }
  let modulationLayout = env["WEETODD_NNC_MODULATION"]=="spans" ? try ModulationSpans(indices:indices.values.map{Int32($0)},tableRows:tableShape[0]) : nil
  inputs.append(try tensor(try data.read("x"),residual:true));inputs.append(try tensor(indices,integer:true))
  let c=try data.read("cos"),s=try data.read("sin")
  guard c.shape==[rows,1,96],s.shape==c.shape,c.values.allSatisfy({$0.isFinite && abs($0)<=1}),s.values.allSatisfy({$0.isFinite && abs($0)<=1}) else { throw ProbeError.invalid("Invalid rotary fixture") }
  for imaginary in [false,true] {
   var values=[Float](repeating:0,count:rows*128)
   for row in 0..<rows {
    for i in 0..<48 {
     guard c.values[row*96+i]==c.values[row*96+i+48],s.values[row*96+i]==s.values[row*96+i+48] else { throw ProbeError.invalid("Rotary halves differ") }
     values[row*128+2*i+(imaginary ? 1 : 0)]=imaginary ? s.values[row*96+i] : c.values[row*96+i]
    }
    if !imaginary { for i in stride(from:96,to:128,by:2) { values[row*128+i]=1 } }
   }
   inputs.append(try tensor(FloatTensor(shape:[rows,1,128],values:values)))
  }
  for index in start..<(start+count) {
   for j in 0..<6 {
    let value=try data.read("block\(index).mod\(j)")
    guard value.shape==tableShape else { throw ProbeError.invalid("Invalid modulation table") }
    inputs.append(try tensor(value))
   }
  }
  if env["WEETODD_NNC_RESIDENCY"]=="block" {
   guard serving else { throw ProbeError.invalid("Block residency uses the persistent worker protocol") }
   let bounded=try BlockResidentStack(graph:graph,weights:weights,adapter:lora,rows:rows,start:start,count:count)
   defer { bounded.unload() }
   try serveStack(model:nil,graph:graph,stream:stream,inputs:&inputs,rows:rows,tableRows:tableShape[0],half:half,loadSeconds:0,start:start,count:count,bounded:bounded)
   return
  }
  let symbols=inputs.map{_ in Input()};var x:Model.IO=symbols[0].io
  var blocks:[H3Block]=[],outputs:[Model.IO]=[]
  let compileStart=Date()
  for i in 0..<count {
   let b=try H3Block(rows:rows,diagnostic:false,adapter:lora,half:half,blockIndex:start+i,ffnScale:env["WEETODD_NNC_FFN_POLICY"]=="layer-scaled" ? layerFFNScale(start+i) : 2,modulationSpans:modulationLayout?.ranges)
   x=b.model([x,symbols[1]]+Array(symbols[(4+i*6)..<(10+i*6)])+[symbols[2],symbols[3]])
   if serving && env["WEETODD_NNC_PROGRESS"]=="1" && ((i+1)%5==0 || i==count-1) {
    let completed=i+1
    x=x.debug(name:"native_progress_\(completed)") { _,stream in
     stream?.joined()
     let data=try! JSONSerialization.data(withJSONObject:["event":"progress","completed":completed,"total":count])
     FileHandle.standardOutput.write(data);FileHandle.standardOutput.write(Data([10]))
    }
   }
   x=RangeTrace.inspect(x,name:"block\(start+i)",retain:true)
   blocks.append(b)
   if rows<=1024 { outputs.append(x) }
  }
  if outputs.isEmpty { outputs=[x] }
  let model=Model(symbols,outputs);model.testing=true;model.maxConcurrency = .limit(4)
  model.compile(inputs:inputs)
  let compileSeconds=Date().timeIntervalSince(compileStart),loadStart=Date()
  for (i,b) in blocks.enumerated() {
   try b.load(checkpoint:weights,adapter:lora,graph:graph,owner:model)
   FileHandle.standardError.write(Data("loaded block \(start+i)\n".utf8))
  }
  let loadSeconds=Date().timeIntervalSince(loadStart)
  if serving {
   try serveStack(model:model,graph:graph,stream:stream,inputs:&inputs,rows:rows,tableRows:tableShape[0],half:half,loadSeconds:loadSeconds,start:start,count:count,modulationLayout:modulationLayout)
   return
  }
  var samples:[Double]=[]
  try graph.withNoGrad {
   for iteration in 0..<iterations {
    let began=DispatchTime.now().uptimeNanoseconds
    let results=model(inputs:inputs[0],Array(inputs.dropFirst()),streamContext:stream);stream.joined()
    samples.append(Double(DispatchTime.now().uptimeNanoseconds-began)/1e9)
    FileHandle.standardError.write(Data("iteration \(iteration): \(samples.last!)s\n".utf8))
    if iteration==iterations-1 {
     for (j,result) in results.enumerated() {
      let name="block\(rows<=1024 ? start+j : start+count-1)"
      if half {
       let output=DynamicGraph.Tensor<Float>(result).toCPU().rawValue
       try output.withUnsafeBytes { bytes in
        guard bytes.bindMemory(to:Float.self).allSatisfy({$0.isFinite}) else { throw ProbeError.invalid("Nonfinite stack output") }
        try Data(bytes).write(to:folder.appendingPathComponent(name+".f32"),options:.atomic)
       }
      } else {
       let output=DynamicGraph.Tensor<BFloat16>(result).toCPU().rawValue
       try output.withUnsafeBytes { bytes in
        guard bytes.bindMemory(to:UInt16.self).allSatisfy({($0 & 0x7f80) != 0x7f80}) else { throw ProbeError.invalid("Nonfinite stack output") }
        try Data(bytes).write(to:folder.appendingPathComponent(name+".bf16"),options:.atomic)
       }
      }
     }
    }
   }
  }
  var usage=rusage();getrusage(RUSAGE_SELF,&usage)
  let report:[String:Any]=["projections":env["WEETODD_NNC_PROJECTIONS"] ?? "fused","modulation_spans":modulationLayout?.ranges.count ?? 0,"rows":rows,"start":start,"count":count,"precision":half ? "fp16" : "bf16","value_scaling":env["WEETODD_NNC_VALUE_SCALING"] ?? "activation","attention_accumulation":env["WEETODD_NNC_ATTENTION_ACCUMULATION"] ?? "fp16","attention_scaling":env["WEETODD_NNC_ATTENTION_SCALE"] ?? "input","attention_order":env["WEETODD_NNC_ATTENTION_ORDER"] ?? "paired","activation_policy":env["WEETODD_NNC_SWISH"] ?? "native","ffn_policy":env["WEETODD_NNC_FFN_POLICY"] ?? "constant-2","seconds":samples,"load_seconds":loadSeconds,"compile_api_seconds":compileSeconds,"peak_rss_bytes":usage.ru_maxrss,"metal_allocated_bytes":MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? 0]
  try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]).write(to:reportURL,options:.atomic)
 }
}

// Power-of-two range management under qualification; restores the scale in FP32.
// Numeric policy from the inspected H3 implementation, independently expressed.
func layerFFNScale(_ index:Int) -> Float {
 if [0,4,11,31,32].contains(index) { return 4 }
 if [1,7,27,33,34,35,37,38,46].contains(index) { return 8 }
 if [18,30,40,41,42].contains(index) { return 16 }
 if index==47 { return 32 }
 if [13,36,43,44,48,49].contains(index) { return 64 }
 if index==39 { return 512 }
 if index==45 { return 1024 }
 return 2
}
