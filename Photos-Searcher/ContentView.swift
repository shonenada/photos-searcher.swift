//
//  ContentView.swift
//  Photos-Searcher
//
//  Created by shonenada on 2023/4/26.
//

import SwiftUI
import Foundation
import CoreML
import GRDB

class LogTracer {
    var date: Date;

    init() {
        date = Date()
    }

    func start() {
        date = Date()
    }

    func logWithTime(msg: String) {
        let now = Date()
        print("\(now.timeIntervalSince(date)) \(msg)")
        date = now
    }
}

struct ContentView: View {
    let logTracer = LogTracer();
    @State var memeFeature: ([Float], [Float])? = nil;
    @State var tokenizer: BPETokenizer? = nil;
    @State var textEncoder: ClipTextEncoder? = nil;
    @State var imageEncoder: ClipImageEncoder? = nil;
    @State var memeClassification: MEMEClassification? = nil;
    @State var photoFeatures: [String: [Float32]] = [:];
    @State var photos: [String: UIImage] = [:];

    @State var isInitModel: Bool = false;
    @State var isModelReady: Bool = false;
    @State var isFeaturesReady: Bool = false;
    @State var keyword: String = "";
    @State var message: String = "";
    @State var displayImages: [(UIImage, Float32, Bool)] = [];
    @State var displayFeatures: [Float32] = [];
    @State var isSearching = false;

    var body: some View {
        VStack {
            if !isInitModel {
                Button(action: {
                    isInitModel = true
                    DispatchQueue.global(qos: .userInitiated).async {
                        initModels()
                        scanPhotos()
                    }
                }, label: {
                    Text("Scan Photos")
                })
                        .padding(20)
                        .border(.blue, width: 2)
            } else if !isModelReady {
                Text("Scanning photos...")
            } else {
                HStack {
                    Text("Keyword")
                    TextField("Keyword", text: $keyword)
                }
                        .padding(20)
                Button(action: {
                    isSearching = true
                    DispatchQueue.global(qos: .userInitiated).async {
                        search()
                    }
                }, label: {
                    if isSearching {
                        Text("Searching...")
                    } else {
                        Text("Search")
                    }
                })
                        .disabled(isSearching || !isModelReady)
                Spacer()

                ScrollView {
                    ForEach(Array(displayImages.enumerated()), id: \.offset) { idx, ele in
                        let (image, probes, meme) = ele
                        Image(uiImage: image)
                                .resizable()
                                .imageScale(.large)
                                .aspectRatio(contentMode: .fit)
                        Text("Probes: \(probes); IsMeme \(meme.description)")
                    }
                }
            }
        }
                .padding()
    }

    func initModels() {
        _initMEMEClassification()
        _initTextEncoder()
        _initImageEncoder()
    }
    
    func _initMEMEClassification() {
        if let _ = self.memeClassification {
            return
        }
        do {
            let config = MLModelConfiguration()
            self.memeClassification = try MEMEClassification(configuration: config)
        } catch {
            print("Failed to init MEME Classification")
            print(error)
        }
        print("Initialized MEME Classification")
    }

    func _initImageEncoder() {
        if let _ = self.imageEncoder {
            return
        }

        // Initialize Image Encoder
        do {
            let config = MLModelConfiguration()
            self.imageEncoder = try ClipImageEncoder(configuration: config)
        } catch {
            print("Failed to init ImageEncoder")
            print(error)
        }
        logTracer.logWithTime(msg: "Initialized image encoder")
        print("Initialized ImageEncoder")
    }

    func _initTokenizer() {
        if let _ = self.tokenizer {
            return
        }

        // Initialize Tokenizer
        guard let vocabURL = Bundle.main.url(forResource: "vocab", withExtension: "json") else {
            fatalError("BPE tokenizer vocabulary file is missing from bundle")
        }
        guard let mergesURL = Bundle.main.url(forResource: "merges", withExtension: "txt") else {
            fatalError("BPE tokenizer merges file is missing from bundle")
        }

        do {
            self.tokenizer = try BPETokenizer(mergesAt: mergesURL, vocabularyAt: vocabURL);
        } catch {
            print(error)
        }
    }

    func _initTextEncoder() {
        _initTokenizer()

        if let _ = self.textEncoder {
            return
        }

        // Initialize Text Encoder
        do {
            let config = MLModelConfiguration()
            self.textEncoder = try ClipTextEncoder(configuration: config)
        } catch {
            print("Failed to init TextEncoder")
            print(error)
        }
        logTracer.logWithTime(msg: "Initialized text encoder")
        print("Initialized TextEncoder")
    }

    func scanPhotos() {
        let startTime = Date()

        let group = DispatchGroup()
        let queue = DispatchQueue.global(qos: .background)

        if (isModelReady) {
            return
        }
        print("Scan photos")
        // Getting features from example photos
        var features: [String: [Float32]] = [:];
        do {
            try dbQueue.read { db in
                let allFeatures = try! Feature.fetchAll(db)
                print("Reload from database: \(Date().timeIntervalSince(startTime))")

                for i in 0..<allFeatures.count {
                    let f = allFeatures[i]
                    let image = f.image
                    let feature = f.feature
                    features[image] = extractArray(from: feature)
                }
            }
        } catch {
            print(error)
        }
        print("Parse vector from database: \(Date().timeIntervalSince(startTime))")

        for j in 0..<1 {
            for i in 0...9 {
                group.enter()
                queue.async {
                    let number = j * 10 + i
                    let name = "image_\(number)"
                    if let feature = features[name] {
//                        print("Load photo \(number) from database")
                        let imageName = "photo\(i).jpg"
                        let image = UIImage(named: imageName)
                        self.photos[name] = image!;
                        self.photoFeatures[name] = feature;
                    }
                    // TODO: Refactor: split codes.
                    print("Scanning photo \(number)")
                    let imageName = "photo\(i).jpg"

                    let image = UIImage(named: imageName)
                    self.photos[name] = image!;

                    let resized = image?.resize(size: CGSize(width: 244, height: 244))
                    // TODO: Use Vision package to resize and center crop image.
                    let ciImage = CIImage(image: resized!)
                    let cgImage = convertCIImageToCGImage(inputImage: ciImage!)
                    do {
                        let input = try ClipImageEncoderInput(imageWith: cgImage!)
                        let output = try self.imageEncoder?.prediction(input: input)
                        let outputFeatures = output!.features
                        let featuresArray = convertMultiArray(input: outputFeatures)
                        self.photoFeatures[name] = featuresArray;

                        try dbQueue.write { db in
                            let featureData = createData(from: featuresArray)
                            try db.execute(sql: "INSERT INTO feature (image, feature) VALUES (?, ?)", arguments: ["image_\(number)", featureData])
                        }
                    } catch {
                        print("Failed to encode image photo \(number)")
                        print(error)
                    }
                    group.leave()
                }
            }

            for k in 0...13 {
                group.enter()
                queue.async {
                    let number = j * 10 + k
                    let name = "meme_\(number)"
                    if let feature = features[name] {
                        print("Load photo \(number) from database")
                        let imageName: String;
                        if k >= 2 {
                            imageName = "meme\(k).gif"
                        } else {
                            imageName = "meme\(k).jpg"
                        }
                        let image = UIImage(named: imageName)
                        self.photos[name] = image!;
                        self.photoFeatures[name] = feature;
                    }
                    // TODO: Refactor: split codes.
                    print("Scanning meme \(number)")
                    let imageName: String;
                    if k >= 2 {
                        imageName = "meme\(k).gif"
                    } else {
                        imageName = "meme\(k).jpg"
                    }

                    let image = UIImage(named: imageName)
                    self.photos[name] = image!;

                    let resized = image?.resize(size: CGSize(width: 244, height: 244))
                    // TODO: Use Vision package to resize and center crop image.
                    let ciImage = CIImage(image: resized!)
                    let cgImage = convertCIImageToCGImage(inputImage: ciImage!)
                    do {
                        let input = try ClipImageEncoderInput(imageWith: cgImage!)
                        let output = try self.imageEncoder?.prediction(input: input)
                        let outputFeatures = output!.features
                        let featuresArray = convertMultiArray(input: outputFeatures)
                        self.photoFeatures[name] = featuresArray;

                        try dbQueue.write { db in
                            let featureData = createData(from: featuresArray)
                            try db.execute(sql: "INSERT INTO feature (image, feature) VALUES (?, ?)", arguments: ["image_\(number)", featureData])
                        }
                    } catch {
                        print("Failed to encode image photo \(number)")
                        print(error)
                    }
                }
                group.leave()
            }
        }
        group.notify(queue: .main) {
            isModelReady = true
            let elapsed = Date().timeIntervalSince(startTime)
            print("Elapsed: \(elapsed)")
        }
    }

    func get_meme_feature() -> ([Float], [Float])? {
        if let featureArray = self.memeFeature {
            return featureArray
        }
        let shape1x77 = [1, 77] as [NSNumber]
        guard let memeTokenArr = try? MLMultiArray(shape: shape1x77, dataType: .float32) else {
            return nil;
        }
        guard let notMemeTokenArr = try? MLMultiArray(shape: shape1x77, dataType: .float32) else {
            return nil;
        }
        do {
            let (_, memeTokenID) = tokenizer!.tokenize(input: "This is a meme image")
            let (_, notMemeTokenID) = tokenizer!.tokenize(input: "This is NOT a meme image")
            for (idx, tokenID) in memeTokenID.enumerated() {
                let key = [0, idx] as [NSNumber]
                memeTokenArr[key] = tokenID as NSNumber
            }
            for (idx, tokenID) in notMemeTokenID.enumerated() {
                let key = [0, idx] as [NSNumber]
                notMemeTokenArr[key] = tokenID as NSNumber
            }

            let input1 = ClipTextEncoderInput(text: memeTokenArr)
            let output1 = try self.textEncoder!.prediction(input: input1)
            let memeFeature = convertMultiArray(input: output1.features)

            let input2 = ClipTextEncoderInput(text: notMemeTokenArr)
            let output2 = try self.textEncoder!.prediction(input: input2)
            let notMemeFeature = convertMultiArray(input: output2.features)

            self.memeFeature = (memeFeature, notMemeFeature)
            return self.memeFeature
        } catch {
            print("Failed to parse features of keyword")
            print(error)
        }
        return nil
    }

    func isMemeImageWithCLIP(imageFeature: [Float]) -> Bool {
        let mf = get_meme_feature()!
        let arr = [mf.0, mf.1]
        let sims = (cosineSimilarityMulti(imageFeature, arr))
        let simsnorm = softmax(sims)
//        print("\(simsnorm[0]) \(simsnorm[1]) \(simsnorm[0] > simsnorm[1])")
        return simsnorm[0] > simsnorm[1]
    }

    func isMemeImage(imageFeature: [Float]) -> Bool {
        let imageArray = try! MLMultiArray(imageFeature)
        let input = MEMEClassificationInput(image: imageArray)
        do {
            let result = try self.memeClassification!.prediction(input: input)
            return result.is_meme == 0
        } catch {
            print("Failed to parse meme type")
            print(error)
        }
        return false
    }

    func get_keyword_features(inputKeyword: String) -> MLMultiArray? {
        let shape1x77 = [1, 77] as [NSNumber]
        guard let multiarray1x77 = try? MLMultiArray(shape: shape1x77, dataType: .float32) else {
            return nil;
        }
        do {
            let (_, tokenIDs) = tokenizer!.tokenize(input: inputKeyword.lowercased())
            for (idx, tokenID) in tokenIDs.enumerated() {
                let key = [0, idx] as [NSNumber]
                multiarray1x77[key] = tokenID as NSNumber
            }
            let input = ClipTextEncoderInput(text: multiarray1x77)
            let output = try self.textEncoder!.prediction(input: input)
            return output.features
        } catch {
            print("Failed to parse features of keyword")
            print(error)
        }
        return nil
    }

    func search() {
        let startTime = Date()

        if self.keyword.isEmpty {
            // TODO: return message
            return;
        }
        let features = get_keyword_features(inputKeyword: "\(keyword)")
//        let memeFeatures = get_keyword_features(inputKeyword: "This is a photo of meme")

        let textArr = convertMultiArray(input: features!)
        var sims: [String: Float32] = [:];
        var memes: [String: Bool] = [:];
//        var sims: [Float32] = [];
        for (name, imageFeature) in photoFeatures {
            let isMeme = isMemeImage(imageFeature: imageFeature)
            let out = cosineSimilarity(textArr, imageFeature)
            sims[name] = out
            memes[name] = isMeme
            print("\(name) = \(isMeme)")
//            sims.append(out)
        }
//        let probs = softmax(sims)
//        var simsMap: [(Int, Float32)] = []
//        for i in 0..<probs.count {
//            let prob = probs[i]
//            simsMap.append((i, prob))
//        }
//        simsMap.sort {
//            $0.1 > $1.1
//        }
        let sortedSims = sims.sorted {
            $0.value > $1.value
        }

        displayImages.removeAll()
        for p in sortedSims.prefix(3) {
            if let photo = photos[p.key] {
                let isMeme = memes[p.key]!;
                displayImages.append((photo, p.value, isMeme));
            }
        }
        isSearching = false;
        print("Search \(Date().timeIntervalSince(startTime))")
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
