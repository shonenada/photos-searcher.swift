//
//  Helper.swift
//  Photos-Searcher
//
//  Created by shonenada on 2023/4/26.
//

import Foundation
import CoreML
import SwiftUI

public func convertMultiArray(input: MLMultiArray) -> Array<Float32> {
    var ret: Array<Float32> = Array()
    if let tmp = try? UnsafeBufferPointer<Float32>(input) {
        ret = Array(tmp)
    }
    return ret
}

public func convertCIImageToCGImage(inputImage: CIImage) -> CGImage! {
    let context = CIContext()
    let cgImage: CGImage = context.createCGImage(inputImage, from: inputImage.extent)!
    return cgImage
}

func createData(from array: [Float]) -> Data? {
    return try? NSKeyedArchiver.archivedData(withRootObject: array, requiringSecureCoding: false)
}

func extractArray(from data: Data?) -> [Float] {
    guard let data = data else {
        return []
    }
    let unarchiver = try? NSKeyedUnarchiver(forReadingFrom: data)
    unarchiver?.requiresSecureCoding = false
    unarchiver?.setClass(NSArray.self, forClassName: "NSArray")
    unarchiver?.setClass(NSNumber.self, forClassName: "__NSCFNumber")
    return unarchiver?.decodeObject(forKey: NSKeyedArchiveRootObjectKey) as? [Float] ?? []
}

