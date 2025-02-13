# Speech Emotion Recognition

## Introduction
Recognizing emotions is an important aspect of AI applications. This project focuses on detecting emotions from speech, which can be useful for various real-world applications such as:
- Call centers playing calming music when a caller is angry.
- Smart cars adjusting speed when detecting emotions like anger or fear.

## Dataset
We utilized a combination of four datasets for training:

1. **RAVDESS** - Ryerson Audio-Visual Database of Emotional Speech and Song
   - Contains high-quality recordings from 24 actors of different genders.
   - Includes speech and song with different emotional intensities.
   - [Dataset Link](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)

2. **SAVEE** - Surrey Audio-Visual Expressed Emotion
   - Contains recordings from four male speakers.
   - Emotion classes include anger, disgust, fear, happiness, neutral, sadness, and surprise.
   - [Dataset Link](https://www.kaggle.com/ejlok1/surrey-audiovisual-expressed-emotion-savee)

3. **TESS** - Toronto Emotional Speech Set
   - Contains recordings from two female speakers (young and older).
   - Helps balance gender distribution in the dataset.
   - [Dataset Link](https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess)

4. **CREMA-D** - Crowd-sourced Emotional Multimodal Actors Dataset
   - Contains recordings from actors of diverse ethnicities.
   - Useful for generalization in transfer learning.

## Implementation

### Approach 1 - 2D CNN Model with MFCC Features
- Uses **Mel-frequency cepstral coefficients (MFCC)** as input features.
- Spectrogram representation is used to extract speech patterns.
- 2D convolution filters capture spatial features from spectrograms.
- Features are passed through **four CNN blocks** with batch normalization, ReLU activation, and max pooling.
- The final output layer has **14 neurons with softmax activation**.

### Approach 2 - Emotion Recognition Using Wav2vec 2.0 Embeddings
- Utilizes **Wav2vec 2.0**, a self-supervised learning framework for raw audio.
- Extracts representations using **local and contextualized encoders**.
- Features from **12 transformer blocks** are averaged with trainable weights.
- The extracted embeddings are used with a **2D CNN model** for classification.

## Model Architecture
- Input sizes:
  - **Approach 1**: (30,216) MFCC features.
  - **Approach 2**: (124,768) Wav2vec embeddings.
- 4 CNN blocks, each consisting of:
  - **Conv2D** with 32 filters (kernel size = (4,10)).
  - **Batch Normalization** and **ReLU activation**.
  - **MaxPooling (2,2)** and dropout (0.2).
- Fully connected layers with **64 neurons**.
- Output layer with **14 neurons** (softmax activation).

## Results
| Approach | Accuracy |
|----------|---------|
| 2D CNN Model with MFCC features | **69.44%** |
| Emotion Recognition using Wav2vec 2.0 | **63.11%** |
