import XCTest
import InferenceContracts
@testable import FlashVSRMLX

final class FlashVSRStreamingMemoryTests:XCTestCase {
  func testSmallCanvasUsesStreamingBlockWeightsRatherThanWholeCheckpointResidency()throws {
    let recipe=try NativeFlashVSRRecipe(sourceID:String(repeating:"a",count:64),inputRGBPath:"/qualification/input.rgb",
      inputSHA256:String(repeating:"b",count:64),outputRGBPath:"/qualification/output.rgb",scratchDirectory:"/qualification/scratch",
      modelRoot:"/models/flash",width:64,height:64,frames:3,fpsNumerator:24,fpsDenominator:1,
      temporalCompression:4,variant:.turbo,decoder:.tiny,seed:42,maximumActiveBytes:4*1024*1024*1024)
    let plan=try FlashVSRMemoryPlan(recipe:recipe)
    XCTAssertNoThrow(try plan.validate(maximumActiveBytes:recipe.maximumActiveBytes,availableDiskBytes:UInt64.max,
      physicalMemoryBytes:8*1024*1024*1024))
  }
}
