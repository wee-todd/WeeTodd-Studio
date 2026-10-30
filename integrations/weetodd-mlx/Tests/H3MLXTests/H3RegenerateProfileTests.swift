import XCTest
@testable import H3MLX

final class H3RegenerateProfileTests: XCTestCase {
  func testRuntimeBaseRequiresEitherPayloadQualifiedRef2VAPreparation() throws {
    for sha in ["600307eba6a67388b333aa5a94537a9ea83da65a6f37664ba22cf079e40a1f3b",
      "37488d7799577d40db8e74318311344cb909c38632c0f693c52976a5509811ff"] {
      XCTAssertNoThrow(try H3RegenerateProfile.validateBaseSHA256(sha))
    }
    XCTAssertThrowsError(try H3RegenerateProfile.validateBaseSHA256(String(repeating:"a",count:64)))
    XCTAssertThrowsError(try H3RegenerateProfile.validateBaseSHA256(H3Ref2VABaseConverter.sourceSHA256))
  }
  // Metadata and dimensions are independent literals from the pinned release headers.
  static func fixture(_ kind: String) -> (metadata: [String:String], tensors: [String:H3RegenerateTensor]) {
    var metadata: [String:String]
    if kind == "official" {
      metadata = ["format":"lora", "scale":"alpha == rank for every module (strengths folded)", "dtype":"float32"]
    } else {
      metadata = ["format":"pt", "split":"refiner ranks 0:384 of the rank-576 merge (refiner|turbo|sharp); alpha == rank, strength folded",
        "source":"Efficient-Large-Model/Open-H3-Regenerate-2K lora/open_h3_regenerate_2k_lora.safetensors",
        "converted_layout":"comfyui_minimax_h3", "baked_scale":"1.0",
        "conversion":"qkv block-diag fused (contiguous q|k|v); mlp.fc1 swiglu halves swapped; alpha folded into lora_B; scale 1.0 baked as a flat multiplier"]
      if kind == "r16" {
        metadata["ss_network_dim"] = "16"; metadata["ss_network_alpha"] = "16.0"
        metadata["ss_training_comment"] = "dimension is resized from 1152 to 16; "
      }
    }
    var tensors: [String:H3RegenerateTensor] = [:]
    for block in 0..<52 {
      let prefix: String
      let specs: [(String,Int,Int,Int)]
      if kind == "official" {
        prefix = block < 50 ? "transformer.transformer_blocks.\(block)." : "transformer.token_refiner.refiner_blocks.\(block-50)."
        specs = [("attn.to_q",7168,5376,576),("attn.to_k",7168,5376,576),("attn.to_v",7168,5376,576),
          ("attn.to_out.0",5376,7168,576),("ff.net.0.proj",28672,5376,576),("ff.net.2",5376,14336,576)]
      } else {
        prefix = block < 50 ? "diffusion_model.blocks.\(block)." : "diffusion_model.token_refiner.blocks.\(block-50)."
        specs = [("attn.qkv_proj",21504,5376,kind == "r16" ? 16:1152),
          ("attn.out_proj",5376,7168,kind == "r16" ? 16:384),
          ("mlp.fc1",28672,5376,kind == "r16" ? 16:384),
          ("mlp.fc2",5376,14336,kind == "r16" ? 16:384)]
      }
      for (suffix,rows,columns,rank) in specs {
        let dtype = kind == "official" ? "F32":"BF16"
        tensors[prefix+suffix+".lora_A.weight"] = .init(dtype:dtype,shape:[rank,columns])
        tensors[prefix+suffix+".lora_B.weight"] = .init(dtype:dtype,shape:[rows,rank])
        if kind == "r16" { tensors[prefix+suffix+".alpha"] = .init(dtype:"BF16",shape:[]) }
      }
    }
    return (metadata,tensors)
  }

  func testFullReleaseHeadersAdmitDistinctProfilesAndSharedRefinerTargets() throws {
    for (kind, profile) in [("official",H3RegenerateProfile.publisherThreeStep),
      ("r384",.extractedRefiner384),("r16",.extractedRefiner16)] {
      let f = Self.fixture(kind)
      XCTAssertEqual(try H3RegenerateProfile.admit(metadata:f.metadata,tensors:f.tensors),profile)
      let roots = try profile.sources(for:"diffusion_model.token_refiner.blocks.1.attn.qkv_proj")
      XCTAssertEqual(roots.count,kind == "official" ? 3:1)
      if kind == "official" {
        XCTAssertEqual(roots,["transformer.token_refiner.refiner_blocks.1.attn.to_q",
          "transformer.token_refiner.refiner_blocks.1.attn.to_k","transformer.token_refiner.refiner_blocks.1.attn.to_v"])
      }
    }
  }
  func testIncompleteExtraWrongRankWrongDtypeAndUnknownMetadataReject() throws {
    for kind in ["official","r384","r16"] {
      let f = Self.fixture(kind)
      let key = f.tensors.keys.sorted().first { $0.hasSuffix(".lora_A.weight") }!
      var missing=f.tensors; missing.removeValue(forKey:key)
      XCTAssertThrowsError(try H3RegenerateProfile.admit(metadata:f.metadata,tensors:missing))
      var extra=f.tensors; extra["unsupported.weight"] = .init(dtype:"F32",shape:[1])
      XCTAssertThrowsError(try H3RegenerateProfile.admit(metadata:f.metadata,tensors:extra))
      var wrong=f.tensors; wrong[key] = .init(dtype:"F32",shape:[512,5376])
      XCTAssertThrowsError(try H3RegenerateProfile.admit(metadata:f.metadata,tensors:wrong))
      var metadata=f.metadata; metadata["baked_scale"] = "0.5"
      XCTAssertThrowsError(try H3RegenerateProfile.admit(metadata:metadata,tensors:f.tensors))
    }
  }
  func testCompanionRolesCannotDoubleApplyMergedTurboOrSharp() throws {
    XCTAssertNoThrow(try H3RegenerateProfile.publisherThreeStep.validateCompanions([.sharpCurve]))
    XCTAssertThrowsError(try H3RegenerateProfile.publisherThreeStep.validateCompanions([.sharpCurve,.turbo]))
    XCTAssertThrowsError(try H3RegenerateProfile.publisherThreeStep.validateCompanions([.sharpCurve,.sharpLinear]))
    XCTAssertThrowsError(try H3RegenerateProfile.publisherThreeStep.validateCompanions([.sharpCurve,.sharpCurve]))
    for profile in [H3RegenerateProfile.extractedRefiner384,.extractedRefiner16] {
      XCTAssertNoThrow(try profile.validateCompanions([.turbo,.sharpLinear,.sharpCurve]))
      XCTAssertThrowsError(try profile.validateCompanions([.sharpCurve]))
    }
  }
}
