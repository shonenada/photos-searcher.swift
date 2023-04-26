//
//  ContentView.swift
//  Photos-Searcher
//
//  Created by shonenada on 2023/4/26.
//

import SwiftUI
import Foundation
import CoreML

struct ContentView: View {
    @State var tokenizer: BPETokenizer? = nil;
    @State var textEncoder: ClipTextEncoder? = nil;
    @State var imageEncoder: ClipImageEncoder? = nil;
    @State var photoFeatures: [MLMultiArray] = [];
    @State var photos: [UIImage] = [];

    @State var isInitModel: Bool = false;
    @State var isModelReady: Bool = false;
    @State var isFeaturesReady: Bool = false;
    @State var keyword: String = "";
    @State var message: String = "";
    @State var displayImages: [UIImage] = [];
    @State var displayFeatures: [Float32] = [];
    
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
                Text("Scaning photos...")
            } else {
                HStack{
                    Text("Keyword")
                    TextField("Keyword", text: $keyword)
                }.padding(20)
                Button(action: {
                    search()
                }, label: {
                    Text("Search")
                })
                Text(message)
                Spacer()

                ScrollView {
                    ForEach(Array(displayImages.enumerated()), id: \.offset) { idx, image in
                        Image(uiImage: image)
                            .resizable()
                            .imageScale(.large)
                            .aspectRatio(contentMode: .fit)
                        Text("Probs: \(displayFeatures[idx])")
                    }
                }
                
            }
        }
        .padding()
    }
    
    func initModels() {
        if (isModelReady) {
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
        print("Initialized tokenier")
        
        // Initialize Text Encoder
        do {
            let config = MLModelConfiguration()
            self.textEncoder = try ClipTextEncoder(configuration: config)
        } catch {
            print("Failed to init TextEncoder")
            print(error)
        }
        print("Initialized TextEncoder")
        
        // Initiailize Image Encoder
        do {
            let config = MLModelConfiguration()
            self.imageEncoder = try ClipImageEncoder(configuration: config)
        } catch {
            print("Failed to init ImageEncoder")
            print(error)
        }
        print("Initialized ImageEncoder")
    }
   
    func scanPhotos() {
        if (isModelReady) {
            return
        }
        print("Scan photos")
        // Getting features from example photos
        for i in 0...9 {
            print("Scaning photo\(i)")
            let image = UIImage(named: "photo\(i).jpg")
            self.photos.append(image!);
            let resized = image?.resize(size: CGSize(width: 244, height: 244))
            // TODO: Use Vision package to resize and center crop image.
            let ciImage = CIImage(image: resized!)
            let cgImage = convertCIImageToCGImage(inputImage: ciImage!)
            do {
                let input = try ClipImageEncoderInput(imageWith: cgImage!)
                let output = try self.imageEncoder?.prediction(input: input)
                self.photoFeatures.append(output!.features);
            } catch {
                print("Failed to encode image photo\(i).jpg")
                print(error)
            }
        }
        isModelReady = true
    }

    func get_keyword_features() -> MLMultiArray? {
        let shape1x77 = [1, 77] as [NSNumber]
        guard let multiarray1x77 = try? MLMultiArray(shape: shape1x77, dataType: .float32) else {
            return nil;
        }
        do {
            let (_, tokenIDs) = tokenizer!.tokenize(input: keyword.lowercased())
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
        if self.keyword.isEmpty {
            // TODO: return message
            return;
        }
        let features = get_keyword_features()
        
        let textArr = convertMultiArray(input: features!)
        var sims: [Float32] = [];
        for i in 0...9 {
            let imageArr = convertMultiArray(input: photoFeatures[i])
            let out = cosineSim(A: textArr, B: imageArr)
            sims.append(out)
        }
        let probs = softmax(sims)
        var simsMap: [(Int, Float32)] = []
        for i in 0...9 {
            let prob = probs[i]
            simsMap.append((i, prob))
        }
        simsMap.sort { $0.1 > $1.1 }
        displayImages.removeAll()
        displayFeatures.removeAll()
        for p in simsMap[0..<3] {
            displayImages.append(photos[p.0]);
            displayFeatures.append(p.1);
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
