//
//  Math2.swift
//  Photos-Searcher
//
//  Created by shonenada on 2023/4/26.
//

import Foundation

public func cosineSim(A: [Float32], B: [Float32]) -> Float32 {
    return dot(A: A, B: B) / (magnitude(A: A) * magnitude(A: B))
}

/** Dot Product **/
public func dot(A: [Float32], B: [Float32]) -> Float32 {
    var x: Float32 = 0
    for i in 0...A.count-1 {
        x += A[i] * B[i]
    }
    return x
}

/** Vector Magnitude **/
public func magnitude(A: [Float32]) -> Float32 {
    var x: Float32 = 0
    for elt in A {
        x += elt * elt
    }
    return sqrt(x)
}
