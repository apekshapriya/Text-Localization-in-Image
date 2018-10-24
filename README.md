# Project
Text Localization in image, ie detection of "Blocks of Text" from an image.

# Approach
Implemented the paper "Accurate Text Localization in Natural Image with Cascaded Convolutional Text Network"

## Dataset Creation
The dataset is created by following these steps-          
    Input images are RGB images consiting of text within it.<br>
    The textual area in the input image is bounded by bounding boxes using an app called labelimg and has been manually tagged.
    Label images are formed using these tagged input image, making the text area masked as pixel-value 0 and 
    rest of the area as pixel-value 1. So the labelled images have pixel values 0 and 1.
