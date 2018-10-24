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
