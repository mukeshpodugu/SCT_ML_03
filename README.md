# SCT_ML_03
# 🧠 Image Classification Using SVM

This project is a simple image classification system built using a Support Vector Machine (SVM). It takes input from a folder of labeled images and classifies them based on visual features extracted from grayscale image data.

## 📂 Dataset

The model uses a folder-based dataset with the following structure:

seg_train/
├── class1/
│ ├── image1.jpg
│ ├── image2.jpg
│ └── ...
├── class2/
│ └── ...


Each subfolder represents a different class, and the classifier is trained to recognize images from these categories.

## ⚙️ Technologies Used

- Python
- OpenCV
- scikit-learn
- NumPy
- Matplotlib

## 🛠️ How It Works

1. Images are loaded from each class folder.
2. Each image is resized and converted to grayscale.
3. Flattened grayscale pixel data is used as input features.
4. A linear SVM model is trained to classify the images.
5. Accuracy and predictions are evaluated on test data.

## 📊 Output

After training, the model prints a classification report showing precision, recall, and F1-score for each class. You can also visualize sample predictions using the popup images.

## 🖼️ Sample Visualization

The script includes a function to display random test images along with their true and predicted labels:

