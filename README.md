# Project
Text Localization in image, ie detection of "Blocks of Text" from an image.

# Approach
Implemented the paper "Accurate Text Localization in Natural Image with Cascaded Convolutional Text Network"

## Dataset Creation

The dataset is created by following these steps-  

### Raw Data
The dataset consists of images and their corresponding xml files. <br>
Input images are RGB images consiting of text within it. <br>
The xml files consists of four cordinates representing bounding boxes of text area for each image. <br>
    
### Preprocessed Data
Input Images are feeded as it is
Label images are formed using the tagged xml files of input images, making the text area masked as pixel-value 0 and <br>
rest of the area as pixel-value 1. So the labelled images have pixel values 0 and 1.
    
Example- <br>
Input Image: <br>
![alt text](https://github.com/apekshapriya/Text-Localization-in-Image/blob/master/img_input.jpg)
    
Labelled Image: <br>
![alt text](https://github.com/apekshapriya/Text-Localization-in-Image/blob/master/img_11.jpg)
   
## Model Architecture
Takes the block 4's output from the standard vgg-16 network and uses this as
input in further architecture. The following layers is used in the whole architecture:<br>
    1) Block4_pool output of vgg16-net<br>
    2) (3,3), (3,7) and (7,3) convolution is applied on 1 parallely.<br>
    3) The output of above three convolution is added and forms the next layer<br>
    4) (2,2) pooling on 4<br>
    5) (1,1) convolution on 4<br>
    6) (1,1) convolution on 5<br>
    7) up-sampling layer(deconvolution) so as to retain the original image size.<br>
        (6 up-sampling layers has been used sequentially. The last up-sampling layer has activation sigmoid to get<br>
        pixel from 0-1)<br>

The model architecture is shown below:<br>
    ![alt text](https://github.com/apekshapriya/Text-Localization-in-Image/blob/master/model.png)
    
### Optimizer and Loss function
    
The optimizer for gradient descent used is "sgd" with learning rate - 0.01. The loss function used is "binary cross-entropy"

The model is trained and saved as model.hd5 file. The trained model is used to predict the test images. An example of it is shown in the result section given below, <br>

### Results

Test Input Image:<br>

![alt text](https://github.com/apekshapriya/Text-Localization-in-Image/blob/master/test_img.jpeg)

Predicted Image: <br>

![alt text](https://github.com/apekshapriya/Text-Localization-in-Image/blob/master/result_img.png)

Here the predicted image has masked pixels in the text area

### To-dos

Try the model with L1 loss regularizer to increase the validation accuracy.<br>
Try Batch Normalization on the Network architecture.<br>
Create bounding boxes over the 0-pixeleted area in a rectangle form for easy interpretation of result.<br>

### Reference
https://arxiv.org/pdf/1603.09423.pdf<br>
https://keras.io/layers/convolutional/
