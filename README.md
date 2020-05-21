# Emoji Prediction

Making our model based upon stacked LSTM and predicting emojis for given sentences.

## Step - 1 Get Emoji Package



```bash
pip install emoji
```
![](https://github.com/JiteshGupta17/Emoji-Prediction/blob/master/ScreenShots/Step%201.JPG)


## Step - 2 Processing the Custom Dataset

```python
train = pd.read_csv('dataset/train_emoji.csv',header=None)
test = pd.read_csv('dataset/test_emoji.csv',header=None)

XT = train[0]
Xt = test[0]

# Converting labels into one hot vector
YT = to_categorical(train[1])
Yt = to_categorical(test[1])
```

## Step - 3 Using Glove vectors
As the Training Dataset is not so large therefore we used pre-trained glove vectors as corpus and then apply LSTM on this corpus for predicting the emojis for their corresponding text.

Download glove 6B.50d.txt
Making our own embedding dictionary - embedding



## Step - 4 Converting sentences into vectors
These vectors will act as the output of the embedding layer.

Input to the embedding layer will be a 2D-tensor of dimensions - batch_size * Maximum_Length of a Sentence

Output of the embedding layer will be a 3D-tensor of dimensions - batch_size * Maximum_Length of a Sentence * 50 where this 50 denotes the length of vector for each word in glove_vectors

![](https://github.com/JiteshGupta17/Emoji-Prediction/blob/master/ScreenShots/Step%204.JPG)

## Step - 5 Creating Our Stacked LSTM model

``` python
from keras.layers import *
from keras.models import Sequential
```

Stacked LSTM means that there will be more than one LSTM layer, this is done for better results as results of one layer are sent to the other and finally to the dense layer with Activation.

## Step - 6 Making Predictions

![](https://github.com/JiteshGupta17/Emoji-Prediction/blob/master/ScreenShots/Final%20Result.JPG)
