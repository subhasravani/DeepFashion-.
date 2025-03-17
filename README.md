# DeepFashion-.
# Fashion-MNIST Classification with PyTorch

This project demonstrates the classification of Fashion-MNIST images using a Convolutional Neural Network (CNN) implemented in PyTorch.

## Dataset

The project uses the Fashion-MNIST dataset, which consists of 70,000 grayscale images of clothing and accessories, divided into 10 categories. The dataset is loaded from a CSV file named "fashion-mnist_train.csv".

## Model

The model used is a simple CNN with two convolutional layers, two max pooling layers, and two fully connected layers. The ReLU activation function is used for non-linearity.

## Training

The model is trained using the Adam optimizer with a learning rate of 0.001 and the CrossEntropyLoss function. The training process involves the following steps:

1. Loading the data from the CSV file.
2. Preparing the data by normalizing pixel values, one-hot encoding labels, and splitting the data into training and validation sets.
3. Defining the CNN model.
4. Setting up training parameters.
5. Training the model for a specified number of epochs.
6. Saving the trained model.

## Evaluation

The trained model is evaluated on the validation set to assess its performance. The accuracy is calculated and printed.

## Results

The trained CNN model achieved a validation accuracy of 91.75%.

## Usage

To run the project, follow these steps:

1. Ensure that the "fashion-mnist_train.csv" file is in the same directory as the code.
2. Install the necessary libraries, including PyTorch, pandas, and torchvision.
3. Run the code cells in the notebook sequentially.

## Next Steps

* **Hyperparameter Tuning:** Experiment with different hyperparameters to potentially improve the model's performance.
* **Further Evaluation:** Evaluate the model on a held-out test set to obtain a more robust estimate of its generalization performance.
* **Model Deployment:** Deploy the trained model for real-world applications.
