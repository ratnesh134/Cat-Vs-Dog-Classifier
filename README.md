# Dogs vs Cats Classification

This project uses a Convolutional Neural Network (CNN) to classify images of dogs and cats. The dataset is obtained from Kaggle, and TensorFlow/Keras is used to implement the deep learning model.

## Dataset

The dataset contains images of dogs and cats, divided into training and validation sets:
- **Training Data**: 20,000 images
- **Validation Data**: 5,000 images

Dataset URL: [Dogs vs Cats Dataset](https://www.kaggle.com/datasets/salader/dogs-vs-cats)

## Project Workflow

1. **Setup**:
   - Ensure the Kaggle API key (`kaggle.json`) is set up to download the dataset.
   - Download and extract the dataset.

2. **Data Preprocessing**:
   - Normalize the image pixel values to the range `[0, 1]`.
   - Resize all images to a uniform size of 256x256 pixels.

3. **Model Architecture**:
   - **Conv2D Layers**: Three convolutional layers with ReLU activation for feature extraction.
   - **Batch Normalization**: Improves training stability.
   - **MaxPooling2D**: Reduces the spatial dimensions of the feature maps.
   - **Fully Connected Layers**: Two dense layers with Dropout for regularization.
   - **Output Layer**: A single neuron with a sigmoid activation function for binary classification.

4. **Training**:
   - Optimizer: `Adam`
   - Loss Function: `binary_crossentropy`
   - Metrics: `accuracy`
   - Trained for 10 epochs with a batch size of 32.

5. **Evaluation**:
   - Plotted training and validation accuracy and loss curves to analyze performance.
   - Observed signs of overfitting, mitigated using Batch Normalization and Dropout.

6. **Testing**:
   - Tested the model on unseen images to verify predictions for dog and cat images.

## Results

The model achieved:
- Training accuracy: ~96%
- Validation accuracy: ~80%
- Correctly classified unseen dog and cat images.

## Usage

### Prerequisites
Ensure you have the required libraries installed. Refer to the `requirements.txt` file for the complete list.

### Steps to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/dogs-vs-cats.git
   cd dogs-vs-cats
2.Set up the Kaggle API key: Place your kaggle.json file in the ~/.kaggle/ directory.

3.Download the dataset:kaggle datasets download -d salader/dogs-vs-cats

4.Extract the dataset:unzip dogs-vs-cats.zip

5.Run the script: Execute the Python script to train the model and test predictions.

## Future Enhancements
Use data augmentation to improve model generalization.

Experiment with different architectures (e.g., ResNet, VGG).

Hyperparameter tuning for better performance.
