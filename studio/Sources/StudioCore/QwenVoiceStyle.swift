import Foundation

public enum QwenVoiceStyle {
  public static let speakers = ["ryan", "aiden", "vivian", "serena", "uncle_fu", "dylan", "eric", "ono_anna", "sohee"]
  public static func label(_ value: String) -> String {
    ["ryan": "Ryan · English", "aiden": "Aiden · English", "vivian": "Vivian · Chinese",
     "serena": "Serena · Chinese", "uncle_fu": "Uncle Fu · Chinese", "dylan": "Dylan · Beijing dialect",
     "eric": "Eric · Sichuan dialect", "ono_anna": "Ono Anna · Japanese", "sohee": "Sohee · Korean"][value] ?? value
  }
  public static let emotions: [(String, String)] = [
    ("Neutral", ""), ("Happy", "Speak in a warm, happy tone."),
    ("Excited", "Speak with excitement and enthusiasm."), ("Sad", "Speak in a quiet, sad tone."),
    ("Angry", "Speak with a firm, angry tone."), ("Whisper", "Speak in a soft whisper."),
    ("Calm", "Speak calmly and reassuringly."), ("Dramatic", "Speak with a dramatic, expressive delivery.")
  ]
}
