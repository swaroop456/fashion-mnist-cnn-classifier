# Fashion-MNIST Image Classification with a CNN

A Convolutional Neural Network (CNN) built in TensorFlow/Keras that classifies grayscale images of clothing into 10 categories, using the Fashion-MNIST dataset.

## Project Overview

Image classification is a core computer vision task used in the real world for things like product tagging, quality inspection, and content moderation. This project builds a CNN that takes a 28x28 grayscale image of a clothing item and predicts which of 10 categories it belongs to.

**Dataset:** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) — 70,000 grayscale images (60,000 train / 10,000 test), 10 classes, loaded directly via `tf.keras.datasets.fashion_mnist`. No manual download needed.

| Label | Class |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Results

| Metric | Value |
|---|---|
| Test Accuracy | 90.85% |
| Test Loss | 0.2448 |
| Best Validation Accuracy | 91.93% (epoch 23) |
| Trainable Parameters | 241,546 |

The weakest-performing class was **Shirt** (~75.5% accuracy) — it's commonly confused with T-shirt/top, Pullover, and Coat, since all four are visually similar upper-body garments at 28x28 resolution.

## Model Architecture

- Conv2D (32 filters, 3x3, ReLU) → MaxPooling2D
- Conv2D (64 filters, 3x3, ReLU) → MaxPooling2D
- Conv2D (128 filters, 3x3, ReLU) → MaxPooling2D
- Flatten → Dense (128, ReLU) → Dropout (0.5) → Dense (10, Softmax)

Trained with data augmentation (rotation, width/height shift, zoom, horizontal flip) and early stopping on validation accuracy (patience = 5).

## Repository Structure

## Author

Swaroop Kumar Vathada
[LinkedIn](https://linkedin.com/in/swaroopkumarvathada) 
