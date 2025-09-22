import os
import cv2
import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
DATASET_PATH = "seg_train"
IMG_SIZE = (64, 64)
SAMPLES_PER_CLASS = 300
labels_map = {}
label_counter = 0
images = []
labels = []
for class_name in sorted(os.listdir(DATASET_PATH)):
    class_path = os.path.join(DATASET_PATH, class_name)
    if not os.path.isdir(class_path):
        continue
    labels_map[label_counter] = class_name
    class_images = [f for f in os.listdir(class_path) if f.endswith(".jpg")]
    class_images = class_images[:SAMPLES_PER_CLASS]
    for img_file in class_images:
        img_path = os.path.join(class_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(gray.flatten())
        labels.append(label_counter)
    label_counter += 1
X = np.array(images)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=list(labels_map.values())))
def show_predictions(n=5):
    idxs = random.sample(range(len(X_test)), n)
    for i in idxs:
        img = X_test[i].reshape(IMG_SIZE)
        pred = labels_map[y_pred[i]]
        true = labels_map[y_test[i]]
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {true} → Predicted: {pred}")
        plt.axis('off')
        plt.show()
show_predictions(5)