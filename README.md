# Audio Language Detection
Our aim is to build a deep learning model to recognize the language of a speaker using audio input format.

## Description
The given dataset contains 10 second of speech recorded in English, German, and Spanish languages.
LibriVox recordings were used to prepare the dataset and particular attention was paid to a big variety of unique speakers since big variance forces the model to concentrate more on language properties than a specific voice.
Samples are equally balanced between languages, genders, and speakers in order not to favour any subgroup.
Finally, the dataset is divided into train and test set. Speakers present in the test set, are not present in the train set. This helps estimate a generalization error.

## Getting Started
The core of the train set is based on 420 minutes (2520 samples) of original recordings. After applying several audio transformations (pitch, speed, and noise) the train set was extended to 12180 minutes (73080 samples).
The test set contains 90 minutes (540 samples) of original recordings. No data augmentation has been applied. Original recordings contain 90 unique speakers. The number of unique speakers was increased by adjusting pitch (8 different levels) and speed (8 different levels). After applying audio transformations there are 1530 unique speakers.

### The dataset is divided into 2 directories:
* train (73080 samples)
* test (540 samples)
### Each sample is an FLAC audio file with:
* sample rate: 22050
* bit depth: 16
* channels: 1
* duration: 10 seconds (sharp)

## Our Approach:

### Data Pre-processing:
We will start by reading all the audio files and converting them into arrays of numbers, utilizing their FLAC format and sample rate.
This will involve extracting the relevant audio features and transforming them into a numerical representation, specifically Mel-frequency cepstral coefficients (MFCC).
### Feature Engineering:
The extracted MFCC features will be used as inputs to train various machine learning algorithms.
We will explore different techniques for feature engineering, such as dimensionality reduction, normalization, and data augmentation, to improve the performance and robustness of our models.
### MFCC:
MFCC (Mel-Frequency Cepstral Coefficients) is a technique used in signal processing to analyze and represent the sound of a human voice or other sound signals.
The sound is first broken down into many tiny pieces called frames, and then for each frame, the MFCC algorithm measures the power of different frequency bands within that frame.
The frequency bands are spaced out in a way that is more like how the human ear perceives sound, which is why it's called Mel-frequency.
Next, the algorithm applies some math operations to these frequency band measurements to reduce the dimensionality and capture the most important features of the sound.
The resulting features are called cepstral coefficients and they can be used as inputs to machine learning models for tasks like speech recognition or music genre classification.
### Model Selection:
We have implemented a deep learning algorithm called convolutional neural networks (CNN) which is the best-performing model for language recognition. 
This will involve experimenting with different hyperparameters, model architectures, and training strategies to optimize the performance of the models.
### Model Evaluation:
We have thoroughly evaluated the performance of the trained models using appropriate evaluation metrics, such as accuracy, precision, recall, and F1-score.
We have achieved an accucary of more than 99% on our test.
This indicates that our CNN model can be deployed to recognize the language of an audio.
