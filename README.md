# FACIAL EXPRESSION RECOGNITION

## Introduction

This project aims to classify the emotion on a person's face into one of seven categories, using support vector machine. This repository is an implementation of this research paper. The model is trained on the JAFFE dataset . This dataset consists of 213 grayscale, 256x256 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised.

## Dependencies

- Python3, [OpenCV](https://opencv.org/), [Scikit-learn](https://scikit-learn.org/stable/)

### Basic Usage

The program will create a window to display the scene capture by webcam.
> Demo

```shell script
python3 main.py
```

You can just use this with the provided pretrained model i have included in the path written in the code file, i have choosen this specificaly since it scores the best accuracy, feel free to choose any but in this case you have to run the later file model_training

> If you just want to run this demo, the following content can be skipped

```shell script
python3 model_training.py
```

## Dataset
I have used [this](https://zenodo.org/record/3451524) dataset

Download JAFFE database and unzip it to dataset folder.

## Evaluate

|Model|Accuracy|
|-----|--------|
|Support vector machine|0.8837|
|Gaussian naive bayes|0.8140|
|Decision tree|0.8140|

## References
1. Lyons, Michael, Kamachi, Miyuki and Gyoba, Jiro. zenodo.org. [Online] 4 14, 1998. https://zenodo.org/record/3451524#.Xtcn-FUzbIU.
2. Rosebrock, Adrian. pyimagesearch.com. [Online] 4 3, 2017. [Cited: 6 3, 2020.] https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/.
3. Happy Bot. stein, joshlam.
4. Shubhrata Gupta, Keshri Verma & Nazil Perveen, Facial Expression Recognition System
Using Facial Characteristic Points And ID3
