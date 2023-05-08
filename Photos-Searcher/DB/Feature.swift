//
//  Feature.swift
//  Photos-Searcher
//
//  Created by shonenada on 2023/4/28.
//

import GRDB

struct Feature {
    var id: Int64?
    var image: String
    var feature: String?
}

extension Feature: Codable, FetchableRecord, MutablePersistableRecord {

    // Mapping purpose
    private enum Columns {
        static let id = Column(CodingKeys.id)
        static let image = Column(CodingKeys.image)
        static let feature = Column(CodingKeys.feature)
    }

    mutating func didInsert(with rowID: Int64, for column: String?) {
        id = rowID
    }

}
