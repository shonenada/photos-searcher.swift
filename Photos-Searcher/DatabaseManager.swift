//
//  DatabaseManager.swift
//  Photos-Searcher
//
//  Created by shonenada on 2023/4/28.
//
import GRDB
import UIKit

var dbQueue: DatabaseQueue!

final class DatabaseManager {
    static func setup(for application: UIApplication) throws {
        let databaseURL = try FileManager.default
            .url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appendingPathComponent("db.sqlite")
        print(databaseURL.path)


        var config = Configuration()
        config.prepareDatabase { db in
            db.trace { print($0) }
        }
        dbQueue = try DatabaseQueue(path: databaseURL.path, configuration: config)
        
        try migrator.migrate(dbQueue)
//        try setupVSS()
    }

    static var migrator: DatabaseMigrator {
        var migrator = DatabaseMigrator()
        
        migrator.registerMigration("createFeature") { db in
            try db.create(table: "feature") { t in
                t.autoIncrementedPrimaryKey("id")
                t.column("image", .text).notNull()
                t.column("feature", .text).notNull()
            }
        }

        return migrator
    }
    
    static func setupVSS() throws {
        try dbQueue.read { db in
//            sqlite3_enable_load_extension(db.sqliteConnection, 1)
//            print(print(Bundle.main.bundlePath))
//            guard var vss0URL = Bundle.main.url(forResource: "vss0", withExtension: "dylib")?.absoluteString else {
//                fatalError("failed to found vss0 url")
//            }
//            if vss0URL.suffix(6) == ".dylib" {
//                vss0URL = String(vss0URL.prefix(vss0URL.count - 6))
//            }
//            print(vss0URL)
//            let output = try! db.execute(sql: """
//            SELECT LOAD_EXTENSION('\(vss0URL)');
//            """
//            )
//            print(output)
            // SELECT LOAD_EXTENSION('\(vss0)');
            // SELECT LOAD_EXTENSION("./vss0");
            
//            sqlite3_enable_load_extension(db.sqliteConnection, 0)
        }
    }
}
