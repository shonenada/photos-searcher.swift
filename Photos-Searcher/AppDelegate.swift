//
//  AppDelegate.swift
//  Photos-Searcher
//
//  Created by shonenada on 2023/4/28.
//

import UIKit

class AppDelegate: UIResponder, UIApplicationDelegate {
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey : Any]? = nil) -> Bool {
        try! DatabaseManager.setup(for: application)
        return true;
    }
}
