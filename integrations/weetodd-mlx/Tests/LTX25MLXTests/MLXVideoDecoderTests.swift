import XCTest
import Foundation
import MLX
import TensorIO
import LTX25Video
@testable import LTX25MLX

final class MLXVideoDecoderTests:XCTestCase {
  func testFiniteScanSupportsTwelveSecondActivationBeyondInt32WithoutFullCopy() throws {
    let shape=[289,96,168,512]
    XCTAssertEqual(shape.reduce(1,*),2_386_427_904)
    XCTAssertGreaterThan(shape.reduce(1,*),Int(Int32.max))
    // A broadcast view has the real failing geometry without a multi-GiB test
    // allocation. The scan must neither flatten nor materialize that whole view.
    let value=broadcast(MLXArray(Float(0)).asType(.bfloat16),to:shape)
    eval(value);Memory.clearCache()
    let baseline=Memory.activeMemory;Memory.peakMemory=baseline
    try MLXVideoDecoder.checkFinite(value)
    XCTAssertLessThan(Memory.peakMemory-baseline,128*1024*1024)
    for frame in [0,144,288] {
      var flags=[Float](repeating:0,count:289);flags[frame] = frame == 144 ? .nan : .infinity
      let invalid=broadcast(MLXArray(flags,[289,1,1,1]).asType(.bfloat16),to:shape)
      XCTAssertThrowsError(try MLXVideoDecoder.checkFinite(invalid))
    }
  }
  func testFiniteScanSplitsOversizedPlanesAndRetainsStridedCoverage() throws {
    for shape in [[1,2,3,5],[2,1,3,5],[2,3,5]] {
      let size=shape.reduce(1,*)
      for index in [0,size/2,size-1] {
        var values=[Float](repeating:0,count:size);values[index] = .nan
        let value=MLXArray(values,shape).transposed()
        XCTAssertThrowsError(try MLXVideoDecoder.checkFinite(value,maximumWindowElements:2))
      }
      XCTAssertNoThrow(try MLXVideoDecoder.checkFinite(MLXArray.zeros(shape).transposed(),maximumWindowElements:2))
    }
    XCTAssertNoThrow(try MLXVideoDecoder.checkFinite(MLXArray(Float(1)),maximumWindowElements:1))
    XCTAssertThrowsError(try MLXVideoDecoder.checkFinite(MLXArray(Float.nan),maximumWindowElements:1))
    XCTAssertNoThrow(try MLXVideoDecoder.checkFinite(MLXArray.zeros([0,3])))
  }
  func testInvalidAllocatorCacheFailsBeforeCheckpointAccess() throws {
    for allowance in [-1,2*1024*1024*1024+1] {
      XCTAssertThrowsError(try MLXVideoDecoder(checkpoint:URL(fileURLWithPath:"/missing/vae"),cacheLimitBytes:allowance)) {
        XCTAssertTrue(String(describing:$0).contains("allocator cache"))
      }
    }
  }
  func testBF16DepthWindowsPreserveDtypeAndMatchUnwindowedTemporalEdges() throws {
    let x=MLXArray((0..<5*3*4*16).map { sin(Float($0)*0.13) },[5,3,4,16]).asType(.bfloat16)
    let weight=MLXArray((0..<16*16*27).map { cos(Float($0)*0.07)*0.01 },[16,16,3,3,3]).asType(.bfloat16)
    let bias=MLXArray.ones([16],dtype:.bfloat16)*Float(0.1)
    for causal in [false,true] { for normalized in [false,true] {
      // Independent whole-input normalization and explicit temporal replication.
      var input=x
      if normalized {
        let norm=MLXFast.rmsNorm(input,weight:.ones([16],dtype:.bfloat16),eps:1e-8)
        input=norm*sigmoid(norm)
      }
      let indices=(-Int(causal ? 2 : 1)..<(5+(causal ? 0 : 1))).map { Int32(max(0,min(4,$0))) }
      let padded=take(input,MLXArray(indices),axis:0)
      var expected:MLXArray?
      for d in 0..<3 {
        let contribution=conv2d(padded[d..<(d+5)],weight[0...,0...,d,0...,0...].transposed(0,2,3,1),padding:1)
        eval(contribution);expected=expected.map { $0+contribution } ?? contribution;eval(expected!)
      }
      let reference=(expected!+bias).asType(.float32).asArray(Float.self)
      for window in [3*4*16,2*3*4*16,100000] {
        let result=try MLXVideoDecoder.convolve(x,weight:weight,bias:bias,causal:causal,normalized:normalized,maximumWindowElements:window)
        XCTAssertEqual(result.dtype,.bfloat16)
        let values=result.asType(.float32).asArray(Float.self)
        for (a,b) in zip(values,reference) { XCTAssertEqual(a,b,accuracy:0.008) }
      }
    } }
    XCTAssertEqual(x.dtype,.bfloat16)
  }
  func testFiniteValidationBoundsScratchAndChecksEveryWindow() throws {
    let input=MLXArray.zeros([32*1024*1024]);eval(input)
    Memory.clearCache();let baseline=Memory.activeMemory;Memory.peakMemory=baseline
    try MLXVideoDecoder.checkFinite(input,maximumWindowElements:65536)
    XCTAssertLessThan(Memory.peakMemory-baseline,16*1024*1024)
    for index in [0,3,4,8] {
      var values=[Float](repeating:0,count:9);values[index] = index%2 == 0 ? .infinity : .nan
      XCTAssertThrowsError(try MLXVideoDecoder.checkFinite(MLXArray(values),maximumWindowElements:4))
    }
    XCTAssertThrowsError(try MLXVideoDecoder.checkFinite(input,maximumWindowElements:0))
  }
  func testMLXAdmissionIncludesWindowWorkspaceAndSingleFrameMinimum() throws {
    var c=VideoDecodeConfiguration();c.maximumActivationBytes=32*1024*1024*1024
    let shape=[1,128,12,24,42]
    let plan=try MLXVideoDecodePlan(shape:shape,configuration:c,precision:.float32)
    let mps=try VideoDecodePlan(shape:shape,configuration:c)
    XCTAssertEqual(plan.outputShape,mps.outputShape)
    XCTAssertGreaterThan(plan.admittedActivationBytes,mps.admittedActivationBytes)
    XCTAssertLessThan(plan.admittedActivationBytes,12*1024*1024*1024)
    c.maximumActivationBytes=plan.admittedActivationBytes
    XCTAssertNoThrow(try MLXVideoDecodePlan(shape:shape,configuration:c,precision:.float32))
    c.maximumActivationBytes -= 1
    XCTAssertThrowsError(try MLXVideoDecodePlan(shape:shape,configuration:c,precision:.float32))
    c.maximumActivationBytes=32*1024*1024*1024
    XCTAssertGreaterThan(try MLXVideoDecodePlan(shape:[1,128,1,64,64],configuration:c,precision:.float32).largestWindowBytes,32*1024*1024)
  }
  func testBF16AdmissionAccountsForGPUAssemblyAndBoundedWorkspace() throws {
    var c=VideoDecodeConfiguration();c.maximumActivationBytes=32*1024*1024*1024
    let shape=[1,128,12,24,42]
    let bf16=try MLXVideoDecodePlan(shape:shape,configuration:c,precision:.bfloat16)
    XCTAssertEqual(try MLXVideoDecodePlan(shape:shape,configuration:c).admittedActivationBytes,bf16.admittedActivationBytes)
    let f32=try MLXVideoDecodePlan(shape:shape,configuration:c,precision:.float32)
    XCTAssertLessThan(bf16.admittedActivationBytes,f32.admittedActivationBytes)
    XCTAssertGreaterThan(bf16.largestWindowBytes,64*1024*1024)
    c.maximumActivationBytes=bf16.admittedActivationBytes-1
    XCTAssertThrowsError(try MLXVideoDecodePlan(shape:shape,configuration:c,precision:.bfloat16))
  }
  func testWindowStorageLivesUntilLastArrayReferenceAndThenReleases() throws {
    weak var storage:AnyObject?
    var output:MLXArray?
    try autoreleasepool {
      let assembly=try MLXVideoWindowBuffer(shape:[1,2,2,1])
      storage=assembly.buffer
      try assembly.append(.ones([1,2,2,1]),start:0)
      output=try assembly.finish()
    }
    XCTAssertNotNil(storage)
    XCTAssertEqual(output!.sum().item(Float.self),4)
    output=nil
    Stream.gpu.synchronize();Memory.clearCache()
    XCTAssertNil(storage,"Managed decoder storage must be released with the last MLX array.")
  }
  func testWindowAssemblyOwnsOneFullOutputAndRejectsGaps() throws {
    let assembly=try MLXVideoWindowBuffer(shape:[16,256,256,16])
    let tile=MLXArray.ones([1,256,256,16]);eval(tile)
    Memory.clearCache();let baseline=Memory.activeMemory;Memory.peakMemory=baseline
    XCTAssertThrowsError(try assembly.finish())
    XCTAssertThrowsError(try assembly.append(tile,start:1))
    for frame in 0..<16 { try assembly.append(tile*Float(frame),start:frame) }
    let output=try assembly.finish()
    // Managed-pointer registration adds the existing output allocation to MLX's
    // accounting. Verify pointer identity and bound only additional workspace.
    let data=output.asData(access:.noCopyIfContiguous).data
    data.withUnsafeBytes { XCTAssertEqual($0.baseAddress,UnsafeRawPointer(assembly.buffer.contents())) }
    XCTAssertLessThan(Memory.peakMemory-baseline-output.nbytes,output.nbytes/2)
    XCTAssertEqual(assembly.allocatedBytes,output.nbytes)
    XCTAssertEqual(output[15,0,0,0].item(Float.self),15)
    XCTAssertThrowsError(try assembly.append(tile,start:16))
    XCTAssertThrowsError(try assembly.finish())
  }
  func testDepthWindowsMatchDirect3DIncludingTemporalEdges() throws {
    let t=3,h=2,w=3,ic=2,oc=3
    let values=(0..<t*h*w*ic).map { sin(Float($0)*0.7) }
    let weights=(0..<oc*ic*27).map { cos(Float($0)*0.31)*0.04 }
    let bias:[Float]=[0.1,-0.2,0.3]
    for causal in [false,true] { for normalized in [false,true] {
      var source=values
      if normalized { for site in 0..<t*h*w {
        let scale=1/sqrt((source[site*ic]*source[site*ic]+source[site*ic+1]*source[site*ic+1])/2+1e-8)
        for c in 0..<ic { let v=source[site*ic+c]*scale;source[site*ic+c]=v/(1+exp(-v)) }
      } }
      var expected:[Float]=[]
      for z in 0..<t { for y in 0..<h { for x in 0..<w { for o in 0..<oc {
        var v=bias[o]
        for i in 0..<ic { for dz in 0..<3 { for dy in 0..<3 { for dx in 0..<3 {
          let zz=max(0,min(t-1,z+dz-(causal ? 2 : 1))),yy=y+dy-1,xx=x+dx-1
          if yy>=0 && yy<h && xx>=0 && xx<w {
            v += source[((zz*h+yy)*w+xx)*ic+i]*weights[(((o*ic+i)*3+dz)*3+dy)*3+dx]
          }
        } } } }
        expected.append(v)
      } } } }
      var baseline:[Float]?
      for window in [h*w*oc,10000] {
        let out=try MLXVideoDecoder.convolve(MLXArray(values,[t,h,w,ic]),weight:MLXArray(weights,[oc,ic,3,3,3]),bias:MLXArray(bias),causal:causal,normalized:normalized,maximumWindowElements:window)
        let actual=out.asArray(Float.self)
        for (a,b) in zip(actual,expected) { XCTAssertEqual(a,b,accuracy:1e-5) }
        if let baseline { for (a,b) in zip(actual,baseline) { XCTAssertEqual(a,b,accuracy:1e-5) } } else { baseline=actual }
      }
    } }
  }
  func testShuffleAndUnpatchPreserveTrainedChannelOrder() throws {
    for (s,t,unpatch) in [(2,2,false),(1,2,false),(2,1,false),(4,1,true)] {
      let shape=[2,2,3,3*s*s*t], values=(0..<shape.reduce(1,*)).map(Float.init)
      let x=MLXVideoDecoder.shuffle(MLXArray(values,shape),spatial:s,temporal:t,unpatch:unpatch)
      var expected:[Float]=[]
      for z in 0..<x.shape[0] { for y in 0..<x.shape[1] { for w in 0..<x.shape[2] { for c in 0..<3 {
        let zz=z+(t>1 ? 1 : 0)
        let channel=unpatch ? ((c*s+w%s)*s+y%s) : (((c*t+zz%t)*s+y%s)*s+w%s)
        expected.append(values[(((zz/t)*shape[1]+y/s)*shape[2]+w/s)*shape[3]+channel])
      } } } }
      XCTAssertEqual(x.asArray(Float.self),expected)
    }
  }
  func testInstalledDecoderAgainstSavedIndependentOracle() throws {
    try installedOracle(precision:.float32,key:"WEETODD_MLX_VIDEO_ORACLE")
  }
  func testInstalledBF16DecoderAgainstPythonAndRestoresWorkspace() throws {
    let previous=getenv("MLX_CONV_WINOGRAD_WORKING_SET").map { String(cString:$0) }
    defer {
      if let previous { setenv("MLX_CONV_WINOGRAD_WORKING_SET",previous,1) }
      else { unsetenv("MLX_CONV_WINOGRAD_WORKING_SET") }
    }
    setenv("MLX_CONV_WINOGRAD_WORKING_SET","536870912",1)
    try installedOracle(precision:.bfloat16,key:"WEETODD_MLX_VIDEO_BF16_ORACLE")
    XCTAssertEqual(String(cString:getenv("MLX_CONV_WINOGRAD_WORKING_SET")),"536870912")
  }
  private func installedOracle(precision:MLXVideoPrecision,key:String) throws {
    let env=ProcessInfo.processInfo.environment
    guard let checkpoint=env["WEETODD_MLX_VIDEO_VAE"],let oracle=env[key] else { throw XCTSkip("Installed decoder oracle is opt-in.") }
    let file=try SafeTensorFile(url:URL(fileURLWithPath:oracle))
    let latent=try MLXWeight.read(file,"latent"),expected=try file.readFloat32(named:"output",maximumBytes:256*1024*1024)
    let url=URL(fileURLWithPath:checkpoint)
    let decoder=try precision == .bfloat16 ? MLXVideoDecoder(checkpoint:url) : MLXVideoDecoder(checkpoint:url,precision:.float32)
    let cacheBefore=Memory.cacheLimit
    var bad=VideoDecodeConfiguration();bad.maximumActivationBytes=1
    XCTAssertThrowsError(try decoder.decode(latent:latent,configuration:bad,receive:{ _ in XCTFail() }))
    XCTAssertEqual(decoder.residentWeightBytes,0)
    var c=VideoDecodeConfiguration();c.maximumActivationBytes=2*1024*1024*1024
    var lowWeight=c;lowWeight.maximumWeightBytes=128*1024*1024
    XCTAssertThrowsError(try decoder.decode(latent:latent,configuration:lowWeight,progress:{ _,_ in XCTFail("Weight admission must precede all layers.") },receive:{ _ in XCTFail() }))
    XCTAssertEqual(decoder.residentWeightBytes,0)
    XCTAssertThrowsError(try decoder.decode(latent:latent,configuration:c,progress:{ _,_ in throw CancellationError() },receive:{ _ in XCTFail() }))
    XCTAssertEqual(decoder.residentWeightBytes,0)
    XCTAssertEqual(Memory.cacheLimit,cacheBefore)
    // The successful decode below reuses the same instance after cancellation.
    var actual:[Float]=[];Memory.peakMemory=0;let start=Date()
    try decoder.decode(latent:latent,configuration:c,receive:{ actual += $0.rgb })
    let seconds=Date().timeIntervalSince(start)
    XCTAssertEqual(actual.count,expected.count)
    var error=0.0,maxabs:Float=0
    for (a,b) in zip(actual,expected) { let d=a-b;error += Double(d*d);maxabs=max(maxabs,abs(d)) }
    let rmse=sqrt(error/Double(actual.count))
    print("MLX_VIDEO seconds=\(seconds) maxabs=\(maxabs) rmse=\(rmse) peak_mlx=\(Memory.peakMemory)")
    XCTAssertLessThan(maxabs,0.01);XCTAssertLessThan(rmse,0.001)
    XCTAssertEqual(decoder.residentWeightBytes,0)
    XCTAssertEqual(Memory.cacheLimit,cacheBefore)
  }
}
