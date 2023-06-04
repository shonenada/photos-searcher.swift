import os
import pickle

import clip
import coremltools
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from skimage.transform import resize
from skimage.io import imread
from PIL import Image


train_path = './content/data/autotrain-data-meme-classification/raw/image_folders/auto/train'


def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clipmodel, preprocess = clip.load("ViT-B/32", device=device)

    labels_dict = {
        'meme': 0,
        'not_meme': 1,
    }

    images = []
    labels = []

    train_meme_path = os.path.join(train_path, 'meme')
    train_not_meme_path = os.path.join(train_path, 'not_meme')

    with torch.no_grad():
        for img_path in os.listdir(train_meme_path):
            img = preprocess(Image.open(os.path.join(train_meme_path, img_path))).unsqueeze(0).to(device)
            features = clipmodel.encode_image(img).numpy().flatten()
            images.append(features)
            labels.append(labels_dict['meme'])

        for img_path in os.listdir(train_not_meme_path):
            img = preprocess(Image.open(os.path.join(train_not_meme_path, img_path))).unsqueeze(0).to(device)
            features = clipmodel.encode_image(img).numpy().flatten()
            images.append(features)
            labels.append(labels_dict['not_meme'])

    flat_data = np.array(images)
    target = np.array(labels)
    df = pd.DataFrame(flat_data)
    df['Target'] = target
    df

    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=77, stratify=y)
    print('Splitted Successfully')

    model = svm.SVC(probability=True, gamma='auto')
    print("The training of the model is started, please wait for while as it may take few minutes to complete")
    model.fit(x_train, y_train)
    print('The Model is trained well with the given images')
    # model.best_params_

    y_pred = model.predict(x_test)
    print("The predicted Data is :", y_pred)

    pickle.dump(model, open('meme_svm_model.p','wb'))
    return model


model = train_model()
coreml_svm = coremltools.converters.sklearn.convert(model, 'image', 'is_meme')

coreml_svm.save('MEMEClassification.mlmodel')
