# Tensorflow certification exam from coursera

This repo contains the code of my study/review journey with coursera tensorflow developer course. <br>
Instead of using Google colab like the course, I run python files and notebook locally to replicate the testing environment since the exam require test taker to use PyCharm.

The code contains in this repo is largly the same as the notebooks within the course. The main difference is the path when using the flow_from_directoy function.

## Courses
- Course 1 - Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning
- Course 2 - Convolutional Neural Networks in TensorFlow
- Course 3 - Natural Language Processing in TensorFlow
- Course 4 - Sequences, Time Series and Prediction

### To train model with gpu locally run this to limit gpu vram usage
```
#GPU memory allocation
cuda does not release the gpu memory when a model is finished training within an ipynb notebook.
Please set a limit for gpu memory allocation so you can train model simultaneously with a python script.


#Limit GPU vram usage to 5gb
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
    except RuntimeError as e:
        print(e)
```
