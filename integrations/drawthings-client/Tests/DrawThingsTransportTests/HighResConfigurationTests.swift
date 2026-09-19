import DataModels
import Foundation
import XCTest
@testable import DrawThingsTransport

final class HighResConfigurationTests: XCTestCase {
  var request: [String: Any] {
    ["modelID": "ltx_2.3_22b_dev_q8p.ckpt", "operation": "video", "inputs": [], "loras": [],
     "configuration": ["width": 2048, "height": 1152, "steps": 8, "seed": 42, "sampler": 17,
                       "guidanceScale": 1, "shift": 6.5, "numFrames": 121, "fps": 25,
                       "hiresFix": true, "hiresFixWidth": 1024, "hiresFixHeight": 576, "hiresFixStrength": 0.5]]
  }

  func testHighResPixelSettingsReachFlatbufferAndEstimate() throws {
    let config = try Configuration.resolve(request)
    XCTAssertEqual(config.sampler, .uniPCTrailing)
    XCTAssertEqual(SamplerType.uniPCTrailing.rawValue, 17)
    XCTAssertTrue(config.hiresFix)
    XCTAssertEqual(config.startWidth, 32)
    XCTAssertEqual(config.startHeight, 18)
    XCTAssertEqual(config.hiresFixStartWidth, 16)
    XCTAssertEqual(config.hiresFixStartHeight, 9)
    XCTAssertEqual(config.hiresFixStrength, 0.5, accuracy: 0.0001)
    XCTAssertEqual(config.steps, 8)
    XCTAssertEqual(config.shift, 6.5)
    let result = try ComputeEstimate.evaluate(request)
    let effective = try XCTUnwrap(result["configuration"] as? [String: Any])
    XCTAssertEqual(effective["hiresFix"] as? Bool, true)
    XCTAssertEqual(effective["hiresFixWidth"] as? Int, 1024)
    XCTAssertEqual(effective["hiresFixHeight"] as? Int, 576)
    XCTAssertEqual(effective["hiresFixStrength"] as? Double ?? 0, 0.5, accuracy: 0.0001)
  }

  func testHighResInvalidControlsCannotSilentlyBecomeDefaults() {
    for (key, bad): (String, Any) in [("hiresFix", 1), ("hiresFix", "true"), ("hiresFixWidth", 1025),
                                     ("hiresFixWidth", true), ("hiresFixHeight", 640),
                                     ("hiresFixStrength", 1.1), ("hiresFixStrength", true), ("stage2Steps", 10)] {
      var value = request
      var fields = value["configuration"] as! [String: Any]
      fields[key] = bad; value["configuration"] = fields
      XCTAssertThrowsError(try Configuration.resolve(value), key)
    }
    for key in ["hiresFixWidth", "hiresFixHeight"] {
      var value = request
      var fields = value["configuration"] as! [String: Any]
      fields.removeValue(forKey: key); value["configuration"] = fields
      XCTAssertThrowsError(try Configuration.resolve(value))
    }
  }

  func testOnePointFiveScaleAndDisabledHighResArePreserved() throws {
    var value = request
    var fields = value["configuration"] as! [String: Any]
    fields["width"] = 1536; fields["height"] = 864
    // Both output dimensions must stay on the 64-pixel grid.
    fields["hiresFixHeight"] = 512; fields["height"] = 768
    value["configuration"] = fields
    XCTAssertTrue(try Configuration.resolve(value).hiresFix)
    fields["hiresFix"] = false; fields.removeValue(forKey: "hiresFixWidth"); fields.removeValue(forKey: "hiresFixHeight")
    value["configuration"] = fields
    XCTAssertFalse(try Configuration.resolve(value).hiresFix)
  }

  func testImageConfigurationsDoNotGainHighResDefaults() throws {
    let value: [String: Any] = ["modelID": "flux_2_klein_4b_q8p.ckpt", "operation": "image", "inputs": [], "loras": [], "configuration": ["width": 512, "height": 512, "steps": 4, "seed": 42]]
    let effective = try XCTUnwrap(ComputeEstimate.evaluate(value)["configuration"] as? [String: Any])
    XCTAssertNil(effective["hiresFix"])
    var bad = value
    bad["configuration"] = ["width": 512, "height": 512, "steps": 4, "seed": 42, "hiresFix": true]
    XCTAssertThrowsError(try Configuration.resolve(bad))
  }
}
