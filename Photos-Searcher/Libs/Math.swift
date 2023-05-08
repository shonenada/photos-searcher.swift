//
//  Math2.swift
//  Photos-Searcher
//
//  Created by shonenada on 2023/4/26.
//

import Accelerate

func cosineSimilarity(_ vector1: [Float], _ vector2: [Float]) -> Float? {
    guard vector1.count == vector2.count else {
        return nil
    }

    var dotProduct: Float = 0.0
    var vector1Norm: Float = 0.0
    var vector2Norm: Float = 0.0

    let count = vDSP_Length(vector1.count)

    vDSP_dotpr(vector1, 1, vector2, 1, &dotProduct, count)
    vDSP_svesq(vector1, 1, &vector1Norm, count)
    vDSP_svesq(vector2, 1, &vector2Norm, count)

    let result = dotProduct / (sqrt(vector1Norm) * sqrt(vector2Norm))

    return result
}
