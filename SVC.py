from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

(xTreino, yTreino), (xTeste, yTeste) = cifar10.load_data()

xTreino_small, _, yTreino_small, _ = train_test_split(xTreino, yTreino, train_size=0.2, stratify=yTreino, random_state=42)

xTeste_small, _, yTeste_small, _ = train_test_split(xTeste, yTeste, train_size=0.2, stratify=yTeste, random_state=42)

XTreino_reshaped = xTreino_small.reshape(xTreino_small.shape[0], -1)
XTeste_reshaped = xTeste_small.reshape(xTeste_small.shape[0], -1)

XTreino_Normalized = XTreino_reshaped / 255.0
XTeste_Normalized = XTeste_reshaped / 255.0

YTreinoFlet = yTreino_small.flatten()
YTesteFlet = yTeste_small.flatten()

n10 = ["airplane", "automobile", "bird", "cat", "deer",
       "dog", "frog", "horse", "ship", "truck"]

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(XTreino_Normalized, YTreinoFlet)

previsto10 = clf.predict(XTeste_Normalized)

print(classification_report(YTesteFlet, previsto10, target_names=n10))
