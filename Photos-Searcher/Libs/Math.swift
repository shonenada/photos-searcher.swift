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

func cosineSimilarityMulti(_ vector1: [Float], _ vector2: [[Float]]) -> [Float] {
    var result = [Float]()
    for vec in vector2 {
        var dotProduct: Float = 0.0
        var mag1: Float = 0.0
        var mag2: Float = 0.0
        
        if vector1.count != vec.count {
            result.append(Float.nan)
        } else {
            for i in 0..<vector1.count {
                dotProduct += vector1[i] * vec[i]
                mag1 += pow(vector1[i], 2)
                mag2 += pow(vec[i], 2)
            }
            let magProduct = sqrt(mag1 * mag2)
            if magProduct == 0 {
                result.append(Float.nan)
            } else {
                result.append(dotProduct / magProduct)
            }
        }
    }
    return result
}
//
//func softmax(_ input: [Float]) -> [Float] {
//    let maxInput = input.max()!
//    var output = [Float](repeating: 0, count: input.count)
//    var expBuffer = [Float](repeating: 0, count: input.count)
//    let inputMinusMax = input.map { $0 - maxInput }
//    vDSP_vexp(inputMinusMax, 1, &expBuffer, 1, vDSP_Length(input.count))
//    let expSum = expBuffer.reduce(0,+)
//    vDSP_vsdiv(expBuffer, 1, &expSum, &output, 1, vDSP_Length(input.count))
//    return output
//}

//func transpose(_ matrix: [[Float]]) -> [[Float]] {
//    let rowCount = matrix.count
//    let colCount = matrix[0].count
//    var transposedMatrix = [[Float]](repeating: [Float](), count: colCount)
//    
//    for col in 0..<colCount {
//        for row in 0..<rowCount {
//            transposedMatrix[col].append(matrix[row][col])
//        }
//    }
//    
//    return transposedMatrix
//}
//
//func acc_transpose(_ matrix: [[Float]]) -> [[Float]] {
//    let rows = matrix.count
//    let columns = matrix[0].count
//    
//    var result = [[Float]](repeating: [Float](repeating: 0, count: rows), count: columns)
//    
//    matrix.withUnsafeBytes { matrixPointer in
//        result.withUnsafeMutableBytes { resultPointer in
//            vDSP_mtrans(matrixPointer.baseAddress!.assumingMemoryBound(to: Float.self),
//                        1,
//                        resultPointer.baseAddress!.assumingMemoryBound(to: Float.self),
//                        1,
//                        vDSP_Length(rows),
//                        vDSP_Length(columns))
//        }
//    }
//    
//    return result
//}
