import Foundation
import XCTest
@testable import ImageLayout

final class RegionalLayoutTests: XCTestCase {
  func testSharedStyleDoesNotLeakUnlistedOccupationsOrAnimals() {
    XCTAssertFalse(MOSSRegionalLayout.styleOnly("A red-haired violinist plays a violin. Warm cinematic lighting.").contains("violinist"))
    XCTAssertFalse(MOSSRegionalLayout.styleOnly("A lion roars at a zebra. Soft lighting.").contains("lion"))
  }
  func testSubjectOwnershipAndLetteringHaveIndependentPrompts() throws {
    let person = ImageLayoutRegion(type: .character, caption: "Woman in a red coat", bounds: .init(x: 0.5, y: 0.1, width: 0.45, height: 0.85))
    var face = ImageLayoutRegion(type: .object, caption: "Her freckled face", bounds: .init(x: 0.6, y: 0.2, width: 0.1, height: 0.1)); face.parentID = person.id
    let text = ImageLayoutRegion(type: .text, caption: "Gold serif letters", text: "HORIZON", bounds: .init(x: 0.1, y: 0.01, width: 0.8, height: 0.1))
    let background = ImageLayoutRegion(type: .object, caption: "A stone wall", bounds: .init(x: 0, y: 0, width: 1, height: 1))
    let draft = ImageLayoutDraft(width: 1200, height: 1696, caption: "A woman poses in a coat. Warm cinematic lighting. Textured paper.", regions: [background,person,face,text])
    let groups = try MOSSRegionalLayout.groups(draft)
    XCTAssertEqual(groups.map(\.role), [.background,.subject,.object,.lettering])
    XCTAssertFalse(groups[0].prompt.contains("Woman in a red coat")); XCTAssertFalse(groups[0].prompt.contains("woman poses"))
    XCTAssertFalse(groups[0].prompt.contains("HORIZON"))
    XCTAssertTrue(groups[1].prompt.contains("freckled face")); XCTAssertFalse(groups[1].prompt.contains("HORIZON"))
    XCTAssertEqual(groups[1].bounds, person.bounds)
    XCTAssertTrue(groups[1].masks!.contains(face.bounds))
    XCTAssertTrue(groups[3].prompt.contains("HORIZON")); XCTAssertFalse(groups[3].prompt.contains("red coat"))
    XCTAssertTrue(groups.allSatisfy { $0.prompt.contains("Warm cinematic lighting") })
  }
  func testFaceAndRigidPropsKeepTheirOwnSpatialStreamsAndDoNotInheritAccentColors() throws {
    let person = ImageLayoutRegion(type:.character,caption:"A black-and-white grayscale photograph of a woman in a black coat",bounds:.init(x:0.05,y:0.1,width:0.9,height:0.9))
    var face = ImageLayoutRegion(type:.object,caption:"A grayscale photograph of her face with arched brows",bounds:.init(x:0.4,y:0.2,width:0.2,height:0.2));face.parentID=person.id;face.analysisRole = .face
    let prop = ImageLayoutRegion(type:.object,caption:"A gold numbered seal",bounds:.init(x:0.05,y:0.2,width:0.2,height:0.1))
    var draft=ImageLayoutDraft(width:1200,height:1696,caption:"A monochrome magazine cover. The background is deep navy blue.",regions:[person,face,prop]);draft.summary.palette=["#FF00FF","#FF0000"]
    let groups=try MOSSRegionalLayout.groups(draft)
    let body=try XCTUnwrap(groups.first{$0.regionID==person.id}),detail=try XCTUnwrap(groups.first{$0.role == .detail})
    XCTAssertFalse(body.prompt.contains("arched brows"));XCTAssertTrue(detail.prompt.contains("arched brows"))
    XCTAssertFalse(body.prompt.contains("#FF00FF"));XCTAssertFalse(detail.prompt.contains("#FF0000"))
    XCTAssertTrue(detail.prompt.contains("grayscale"))
    XCTAssertTrue(groups.contains{$0.role == .object && $0.regionID==prop.id})
    XCTAssertTrue(groups[0].prompt.contains("navy-blue background"))
    XCTAssertFalse(body.masks!.contains(person.bounds))
  }
  func testStripedClothesAndLetteringDoNotBecomePhotographicGrayscale() throws {
    let person=ImageLayoutRegion(type:.character,caption:"A color photograph of a woman in a black-and-white striped coat",bounds:.init(x:0.1,y:0.1,width:0.8,height:0.8))
    var child=ImageLayoutRegion(type:.object,caption:"Monochrome patterned clothing",bounds:.init(x:0.2,y:0.5,width:0.6,height:0.4));child.parentID=person.id
    let text=ImageLayoutRegion(type:.text,caption:"Black-and-white sans-serif lettering",text:"TITLE",bounds:.init(x:0,y:0,width:1,height:0.1))
    let draft=ImageLayoutDraft(width:1024,height:1024,caption:"Color graphic design",regions:[child,person,text])
    let groups=try MOSSRegionalLayout.groups(draft)
    XCTAssertFalse(groups.first{$0.role == .subject}!.prompt.contains("grayscale"))
    XCTAssertFalse(groups.first{$0.role == .lettering}!.prompt.contains("photography"))
    var gray=draft;gray.regions[1].caption="A grayscale photograph of a woman";
    XCTAssertTrue(try MOSSRegionalLayout.groups(gray).first{$0.role == .subject}!.prompt.contains("grayscale"))
  }
}
