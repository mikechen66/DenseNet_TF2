# DenseNet_TF2

The DenseNet is a type of conv neural network that utilises dense connections between layers, 
through Dense Blocks, where the authors connect all layers (with matching feature-map sizes) 
directly with each other. To preserve the feed-forward nature, each layer obtains additional 
inputs from all preceding layers and passes on its own feature-maps to all subsequent layers.

Make the necessary changes to adapt to the new environment of TensorFlow 2.3, Keras 2.4.3, 
CUDA Toolkit 11.0, cuDNN 8.0.1 and CUDA 450.57. In addition, write the new lines of code to 
replace the deprecated code. I would like to thank all of the creators and interptretors for 
their contributions.
