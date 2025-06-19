# Rooftop Detection Project 🚀

Hey there! This project is all about detecting rooftops using some cool deep learning models like Mask R-CNN and Faster R-CNN. The main goal is to boost detection accuracy and play around with different approaches to see what works best.
<br>
Original dataset: https://www.kaggle.com/datasets/ayushbajpai16/rooftop-satellite-imagery-augmented-yolo-format
## Project Structure 📂

Here's a quick rundown of the files in this project:

- `train.py`: This is where the magic happens! It contains the main training loop for the model. 🏋️‍♂️
- `dataset.py`: This file takes care of loading and preprocessing the dataset. Think of it as the data chef! 📊
- `utils.py`: This one's got all the handy utility functions that are used all over the project. 🔧
- `data_preprocessing.py`: Before training, data needs a little makeover, and this file handles all that. ✨
- `configuration.py`: All the nitty-gritty settings for the path you might need are there. ⚙️
- `model.py`: This file is home to the model definitions and other related functions. 🤖
- `config.yaml`: This file has all the important configuration stuff like epochs and learning rate. 📝

## Detailed Description of Files 📜

### `train.py`
This file is like the conductor of an orchestra, managing the training loop. It loads the dataset, initializes the model, and runs the training iterations.

### `dataset.py`
Think of this file as the data butler. It makes sure that the data is nicely prepped and ready for training.

### `utils.py`
This file is packed with all sorts of useful functions that help with tasks like data manipulation, visualization, and logging. It's like the Swiss Army knife of the project.

### `data_preprocessing.py`
Before training, the data needs to be cleaned up and transformed. This file handles all that preprocessing work.

### `configuration.py`
This file is where all the important settings live. Things like learning rate, batch size, and model hyperparameters are tucked away here.

### `model.py`
This file is all about the models. It has the architecture definitions for Mask R-CNN and Faster R-CNN, along with functions for initializing and training these models.

### `config.yaml`
Here's where you'll find all the important config details like epochs and learning rate. It's like the recipe book for training the model.

## Methodology 🧠

I've been playing around with two main approaches for rooftop detection: segmentation masks and bounding boxes. The models I'm using are Mask R-CNN and Faster R-CNN, and I'm using torchvision to implement them. So far, the models are kinda stuck at around 23% mean Average Precision (mAP). I haven't had much time to test the bounding box approach yet, but I'm excited to see how it performs!

### Mask R-CNN
Mask R-CNN is like the fancy cousin of Faster R-CNN. It adds a branch for predicting segmentation masks on each Region of Interest (RoI), which runs side by side with the existing branch for classification and bounding box regression. Pretty neat, huh? 🎭

### Faster R-CNN
Faster R-CNN is like a detective that uses a Region Proposal Network (RPN) to generate region proposals. These proposals are then classified and refined to get the final detection results. 🔍

## Results 📈

Right now, the models are chugging along with around 23% mAP. There's definitely room for improvement, and I'm planning to do some more experimenting and tuning to boost performance. <br>
Graphics will come soon 🛠️

## Future Work 🚀

There's a lot more I want to try out and improve:

1. **Model Architecture**: I'm thinking about playing around with different backbone architectures for the models. Maybe even create a module that can generate models with different backbones. 🏗️
2. **Data Augmentation**: More data augmentation techniques could really help improve the model's ability to generalize. 🔀
3. **Hyperparameter Tuning**: I need to do a deep dive into hyperparameter tuning to get the best performance out of the models. 🎛️
4. **Bounding Box Approach**: I haven't spent much time on the bounding box approach yet, but I'm hoping it might outperform the segmentation mask method. 📦
5. **Implementation of TODOs**: There are a bunch of TODOs scattered throughout the code that I need to tackle to add more features and improvements. 📝

## Conclusion 🏁

This project is my little playground for experimenting with rooftop detection using deep learning. The results so far are okay, but there's definitely a lot more to explore and improve. If you're interested, feel free to dive in and contribute!

Happy coding! 🌟

## References 📚

- Mask R-CNN: [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)
- Faster R-CNN: [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)


<small>This file was improved with my Mistral AI Agent</small>
