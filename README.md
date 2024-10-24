# FishClassificaitonANN
# Fish Species Classification with Deep Learning
- This project focuses on the classification of different fish species using a deep learning approach. A dataset of fish images was used to train a ANN, optimizing its performance through hyperparameter tuning and evaluation on a test dataset.
## Project Overview
- The goal of this project is to classify 9 species of fish using a dataset containing 9000 images. Each species has 1000 images, and the project applies state-of-the-art deep learning techniques to achieve high classification accuracy.
## Key Steps in the Project:
- Data Collection: The dataset consists of images of 9 fish species. Images were preprocessed, including resizing and normalization.
- Data Agumentation: Techniques such as rescaling and validation split were applied using ImageDataGenerator from Keras to enhance model generalization.
- Model Architecture: A Sequential model was built using fully connected Dense layers, dropout for regularization, and a softmax activation for multi-class classification. The model was trained using Adam optimizer and categorical crossentropy loss function.
- Hyperparameter Optimization: To improve the training of the model.
- Learning Rate: Set to 0.00001 for stable convergence; Dropout Rate: A 0.3 dropout was applied to prevent overfitting; Batch Size: Each batch contained 32 samples to optimize training efficiency and stability.
- Early Stopping: The model utilized early stopping to prevent overfitting by stoping training if the validation loss didn’t improve for 3 consecutive epochs.
## Model Evaluation 
The model achieved a test accuracy of ~95% and a test loss of 0.14, indicating effective performance in classifying fish species.

- Please note that the differences in results may be attributed to the fact that when a Kaggle notebook is shared, it is re-run upon submission. As a result, small changes in metrics like accuracy or loss may occur due to factors such as randomness in data splits or model initialization.

## Additional informations abot training results:

#Trial 1
-Number of Epochs: 10
- Model Architecture:
  - Input Layer: Flatten (224x224x3)
  - Dense Layer 1: 512 units, ReLU activation
  - Dense Layer 2: 256 units, ReLU activation
  - Dense Layer 3: 128 units, ReLU activation
  - Output Layer: Softmax activation (9 classes)
- Test Loss: 0.4065
- Test Accuracy: 85%
![image](https://github.com/user-attachments/assets/742551a0-f65c-4f4c-a899-e86606b8f253)
The data overfitting

#Trial 2
- Number of Epochs: 10
- Batch Size: 128
- Model Architecture:
  - Input Layer: Flatten (224x224x3)
  - Dense Layer 1: 512 units, ReLU activation
  - Dense Layer 2: 256 units, ReLU activation
  - Dense Layer 3: 128 units, ReLU activation
  - Output Layer: Softmax activation (9 classes)
- Test Loss: 0.4173
- Test Accuracy: 86%
![image](https://github.com/user-attachments/assets/524cca7f-fcfa-4270-8a2e-fa1ec3cbcf02)

#Trial 3
- Number of Epochs: 10
- Batch Size: 128
- Model Architecture:
  - Input Layer: Flatten (224x224x3)
  - Dense Layer 1: 512 units, ReLU activation
  - Dropout 1: 0.5
  - Dense Layer 2: 256 units, ReLU activation
  - Dropout 2: 0.5
  - Dense Layer 3: 128 units, ReLU activation
  - Output Layer: Softmax activation (9 classes)
- Test Loss: ?
- Test Accuracy: 11%

#Trial 4
- Number of Epochs: 25
- Batch Size: 128
- Model Architecture:
  - Input Layer: Flatten (224x224x3)
  - Dense Layer 1: 256 units, ReLU activation
  - Dropout 1: 0.3
  - Dense Layer 2: 128 units, ReLU activation
  - Dropout 2: 0.3
  - Output Layer: Softmax activation (9 classes)
- Test Loss: 0.3486
- Test Accuracy: 88%
![image](https://github.com/user-attachments/assets/493a5ebf-5f58-4554-a409-771ca963cb7d)
better but overfitting

#Trial 5
- Number of Epochs: 10
- Batch Size: 128
- Model Architecture:
  - Input Layer: Flatten (224x224x3)
  - Dense Layer 1: 256 units, ReLU activation
  - Dense Layer 2: 128 units, ReLU activation
  - Dense Layer 3: 64 units, ReLU activation
  - Output Layer: Softmax activation (9 classes)
- Test Loss: 0.2457
- Test Accuracy: 91%
![image](https://github.com/user-attachments/assets/ed5be623-fd64-486a-9d64-a172af323bb4)
better accuracy but overfitting

#Trial 6
- Number of Epochs: 100
- Batch Size: 32
- Model Architecture:
  - Input Layer: Flatten (224x224x3)
  - Dense Layer 1: 512 units, ReLU activation
  - Dropout 1: 0.3
  - Dense Layer 2: 256 units, ReLU activation
  - Output Layer: Softmax activation (9 classes)
  - Optimization larning rate: 0.00001
- Test Loss: 0.1618
- Test Accuracy: 95%
![image](https://github.com/user-attachments/assets/009465f4-c382-4dea-9653-a65e610e0965)
Best result but after re-reunning the same model could not get same accuracy but get close as you can see in the project file provided (This project file also uploaded but without markdowns, named as fishclassification_best.ipynb).

Keggle link: https://www.kaggle.com/code/karinmanto/fishclassificaiton-ann/notebook

