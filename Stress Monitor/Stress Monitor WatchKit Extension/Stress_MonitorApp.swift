//
//  Stress_MonitorApp.swift
//  Stress Monitor WatchKit Extension
//
//  Created by Joshua Thompson on 4/12/21.
//

import SwiftUI

@main
struct Stress_MonitorApp: App {
    @SceneBuilder var body: some Scene {
        WindowGroup {
            NavigationView {
                ContentView()
            }
        }

        WKNotificationScene(controller: NotificationController.self, category: "myCategory")
    }
}
