import Foundation
import XCTest
@testable import InferenceContracts

final class NativeAccelerationRegistryTests: XCTestCase {
  func testResearchAndImportRecordsCannotBecomeRuntimeCapabilities() throws {
    XCTAssertEqual(Set(NativeAccelerationRegistry.candidates.map(\.id)).count,NativeAccelerationRegistry.candidates.count)
    for candidate in NativeAccelerationRegistry.candidates {
      XCTAssertFalse(candidate.runnable)
      XCTAssertFalse(candidate.directRegenerationLatentsQualified)
      XCTAssertThrowsError(try JSONDecoder().decode(NativeCapabilityDescriptor.self,from:JSONEncoder().encode(candidate)))
      XCTAssertTrue(candidate.sourceRevision.count == 40)
    }
    XCTAssertEqual(NativeAccelerationRegistry.candidates.first{$0.id == "fastgen-h3-pdd"}?.availability,.researchOnly)
    XCTAssertEqual(NativeAccelerationRegistry.candidates.first{$0.id == "pdmd-h3-4nfe-bf16"}?.availability,.importCandidate)
  }
  func testPDMDClocksCountEvaluationsNotTerminalPoints() throws {
    let four=try XCTUnwrap(NativeAccelerationRegistry.candidates.first{$0.id == "pdmd-h3-4nfe-bf16"})
    XCTAssertEqual(four.evaluations,[4]);XCTAssertEqual(four.videoSigmas?.count,5)
    XCTAssertEqual(four.audioSigmas,[1,Double(Float(0.9)),0.75,0.5,0])
    let recommended=try XCTUnwrap(NativeAccelerationRegistry.candidates.first{$0.id == "pdmd-h3-2nfe-bf16-recommended"})
    let paper=try XCTUnwrap(NativeAccelerationRegistry.candidates.first{$0.id == "pdmd-h3-2nfe-bf16-paper"})
    XCTAssertEqual(recommended.videoSigmas,paper.videoSigmas)
    XCTAssertEqual(recommended.audioSigmas,[1,Double(Float(6.0/7)),0])
    XCTAssertEqual(paper.audioSigmas,[1,0.75,0]);XCTAssertNotEqual(recommended.id,paper.id)
    XCTAssertEqual(recommended.heads,.ordinaryH3)
  }
  func testDMADStudentsRequireDistinctNoiseAndWeightIdentities() throws {
    let students=NativeAccelerationRegistry.candidates.filter{$0.id.hasPrefix("dmad-h3-")}
    XCTAssertEqual(students.count,2)
    XCTAssertNotEqual(students[0].asset,students[1].asset)
    for student in students {
      XCTAssertEqual(student.update,.denoiseThenRenoise)
      XCTAssertEqual(student.noise,.initialAndEveryStepVideoThenAudio)
      XCTAssertEqual(student.audioSigmas,[1,Double(Float(6.0/7)),Double(Float(2.0/3)),Double(Float(0.4)),0])
      XCTAssertEqual(student.evaluations,[4])
    }
  }
  func testPDDHeadAdmissionDoesNotAcceptOrdinaryPDMDHeads() throws {
    let pdd=NativeAccelerationHead.h3Parallel32
    try pdd.validate(videoWeight:[3072,5376],audioWeight:[1024,5376])
    XCTAssertThrowsError(try pdd.validate(videoWeight:[96,5376],audioWeight:[32,5376]))
    XCTAssertThrowsError(try pdd.validate(videoWeight:[3072,5376],audioWeight:[32,5376]))
    try NativeAccelerationHead.ordinaryH3.validate(videoWeight:[96,5376],audioWeight:[32,5376])
    XCTAssertEqual(try pdd.outputRow(head:3,baseRow:5,video:true),3*96+5)
    XCTAssertThrowsError(try pdd.outputRow(head:32,baseRow:0,video:true))
  }
  func testDataWardAndSigmaVelocityReachSameEndpoint() throws {
    let x=2.0,u=3.0,sigma=0.8,next=0.2
    let v=u*NativeAccelerationVelocity.dataWard.signToSigmaVelocity
    let native=x+(sigma-next)*u,flow=x+(next-sigma)*v
    XCTAssertEqual(native,flow,accuracy:1e-14)
    XCTAssertEqual(x+sigma*u,x-sigma*v,accuracy:1e-14)
    XCTAssertEqual(NativeAccelerationVelocity.sigma.signToSigmaVelocity,1)
  }
  func testVDNRequiresLearnedBranchAndStageSpecificAdapters() throws {
    for record in NativeAccelerationRegistry.vdnStages {
      let runtime=try NativeCapabilityDescriptor.vdn(evaluations:record.evaluations)
      XCTAssertEqual(record.requiredRoles,runtime.requiredRoles)
      XCTAssertTrue(record.requiredRoles.contains(.linearBranch));XCTAssertTrue(record.requiredRoles.contains(.standardAdapter))
      XCTAssertEqual(record.requiredRoles.contains(.turboAdapter),record.evaluations == 8)
      XCTAssertEqual(runtime.sampling.sigmas.count,record.evaluations+1)
      XCTAssertEqual(runtime.sampling.audioClock,.separateSigma)
      XCTAssertFalse(record.directRegenerationLatentsQualified)
      XCTAssertEqual(record.downstreamRoute,.decodedMovie)
    }
  }
}
