## Project: Follow Me
### Taariq Hassan (thassan743)

---

[//]: # (Image References)

[image1]: ./docs/misc/architecture.png
[image2]: ./docs/misc/loss.png
[image3]: ./docs/misc/follow_target.png
[image4]: ./docs/misc/no_target.png
[image5]: ./docs/misc/distance_target.png

## [Rubric](https://review.udacity.com/#!/rubrics/1155/view) Points

---
### Writeup / README

The focus of this project was to train a deep neural network that can be used on a simulated quadcopter to track a target in the "follow me" mode. In order to achieve this, a fully convolutional network (FCN) is required, since it maintains spatial information, over a fully connected network which is normally used for basic image classification. This is necessary since we need to know where in the image the target person (the "hero") is.

Below I will discuss some of the code used in the implementation of the network, the final network structure that was used, the results achieved, and possible improvements to the project.

#### The Code

The first step in the provided jupyter notebook was to complete the `encoder block`. The final implementation can be seen below:

```
def encoder_block(input_layer, filters, strides):
    
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer
```

The encoder makes use of the provided `separable_conv2d_batchnorm` function which implements a separable convolution layer with batch normalisation.

The next step was to implement the `decoder block`. The purpose of the decoder is to upsample the layers back to the original size of the input and extract the spatial information from the image. The final decoder implementation can be seen below:

```
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    upsampled = bilinear_upsample(small_ip_layer)
    
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    concat = layers.concatenate([upsampled, large_ip_layer])
    
    # TODO Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(concat, filters)
    output_layer = separable_conv2d_batchnorm(output_layer, filters)
    
    return output_layer
```

The first step in the decoder is to upsample the input layer using the provided `bilinear_upsample` function which utilises the `BilinearUpSampling2D` class found in the provided `utils` code [here](https://github.com/thassan743/RoboND-DeepLearning-Project/blob/master/code/utils/separable_conv2d.py). This function upsamples the input by a factor of 2.

The second step implements layer concatenation, which is similar to skip connections. The purpose of this step is to concatenate the previously upsampled layer with another layer that has more spatial information. This improves the performance of the network since it allows the network to use information from previous layers which may have been lost in the encoding process. This step is implemented using the `layers.concatenate` function from `tf.contrib.keras`.

The final step of the decoder is to add additional separable convolutions such that the network is able to extract the extra spatial information from the higher resolution layers concatenated in the previous step. Here, two `separable_conv2d_batchnorm` were added.

#### The Network

Now that we have a completed encoder and decoder, we can build the FCN model. My plan was to keep the model as simple as possible initially to get a baseline for the performance with different hyper-parameters, and then increase the complexity from there if necessary. In the end i was able to achieve the required final score with this model after some adjustments and tuning. This will be discussed in further detail later. The final model implementation can be seen below:

```
def fcn_model(inputs, num_classes):
	
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    l_enc_1 = encoder_block(inputs, 32, 2)
    l_enc_2 = encoder_block(l_enc_1, 64, 2)
    
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    l_1by1_1 = conv2d_batchnorm(l_enc_2, filters = 128, kernel_size = 1, strides = 1)
    
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    l_dec_1 = decoder_block(l_1by1_1, l_enc_1, filters = 64)
    l_dec_2 = decoder_block(l_dec_1, inputs, filters = 32)
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(l_dec_2)
```

The architecture used for the FCN consists of two encoder layers, a 1x1 convolution layer and two decoder layers. There are also two skip connections from Encoder 1 to Decoder 1 and from the input to Decoder 2 which make use of the layer concatenation step of the decoder. The architecture can be seen in the image below. My initial model had the same architecture except that each layer had half the filter depth the final model shown above. With that model, I was able to achieve just over 40%. By increasing the depth, the performance improved by approximately 2%. The network is discussed in more detail below.

![alt text][image1]

#### Encoder

The input to the network is an image of size 160x160x3. This image is passed through 2 encoders. The purpose of an encoder is to extract features from the input image. Having a number of encoder layers after each other allows the network to extract more complex features the deeper the network goes. For this project, I was able to achieve the required result using 2 encoders. The first encoder has a filter depth of 32 and a stride of 2 which results in an output image from the encoder of dimensions 80x80x32. Similarly for encoder 2 which has a filter depth of 64, the output is an image with dimensions of 40x40x64.

#### 1x1 Convolution (vs. Fully Connected Layers)

The use of a 1x1 convolution layer in the network is very important. In conventional CNNs, a fully connected layer is normally used when training a classifier. However, the output of this type of network is a 2D tensor, and therefore all spatial information about the input is lost. By using a 1x1 convolution in an FCN, the output is a 4D tensor and therefore it retains spatial information. By retaining spatial information, we are able to locate specific objects in the image and not just classify objects. This is essential in order to allow a system to locate and follow a specific target in an image.

The 1x1 convolution layer used here has a filter depth of 128. This results in an output image of 40x40x128.

#### Decoder

The purpose of the decoder is to upsample the image. We saw that in the previous layers, the size of the image was significantly smaller than the original input image. However, in order to perform semantic segmentation on the original input image, we need the output to be of the same size as the input. In this way, we can obtain information from each pixel in the image in order to identify an object (the hero) and also the location of the object. The decoder layers used here both have a bilinear upsample factor of 2. Decoder 1 has a filter depth of 64 and includes a skip connection from the output of Encoder 1. The output from this decoder is therefore 80x80x64. Decoder 2 has a filter depth of 32 and a skip connection from the input. The output from this decoder is then 160x160x32. We can see that the shape of the output matches that of the input image.

#### Hyper-Parameters

The Hyper-Parameters used in my final model were as follows:

```
learning_rate = 0.005
batch_size = 64
num_epochs = 20
steps_per_epoch = 200
validation_steps = 50
workers = 8
```
Not much time was spent tuning the hyper-parameters since I was able to reach the required score fairly quickly. That being said, there was 

**learning_rate:** I started with a learning rate of 0.01 and then decreased it to 0.005. Decreasing the learning rate resulted in a slower decrease in the loss curves, requiring a longer training time, but ultimately a lower loss at the end of the training.

**batch_size:** The number of images used for training in a single step. Too large of a batch size may cause the system to run out of memory. I started with a batch size of 64 and didn't need to change it. I did not experience any issues with memory during training on AWS so I could have tried increasing the batch size even more, however there was no need.

**num_epochs:** The number of training cycles to perform. For the number of epochs, I found that after about 20 epochs, there was only a slight increase in performance for the added training time. However, I have seen some other people training for many more epochs and have seen improvements. For my model I left it at 20 and was able to achieve the required IoU score. If I were to try and improve the score, I would probably increase the number of epochs considerably to see what improvements can be achieved.

**steps_per_epoch:** The number of batches to train on in a single epoch. I left this at the default value of 200.

**validation_steps:** The number of batches to use during validation of a single epoch. I also left this at the default value of 50.

**workers:** The number of processes to use. I increased this from 2 to 8 but did not notice any signicant changes in performance or speed. In the end I just left it at 8.

#### Results

Using the above network architecture, I trained the network using an AWS instance. Before collecting any additional data, I trained with the provided data. While the performance with the provided data is not optimal (which will be discussed later), I found that I was able to achieve the required accuracy of 40% without gathering any additional data.

I kept the model fairly simple since initially I was training on my local machine. However, I moved over to AWS as soon as my credits were approved and found that my simple 2 encoder 2 decoder model worked well enough.

With the above model and hyper-parameters, I achieved a **`final_score = 0.426002134326`**. The image below shows the loss curve for the last epoch of training. The final `train_los = 0.0197` and `val_loss = 0.0260`

![alt text][image2]

In the jupyter notebook, predictions are generated from the model for various scenarios. The first scenario is with the drone following the target. In the image below we can see, from left to right, the original image, the ground truth, and the models prediction. We can see that the model performs fairly well in this scenario. This is backed up by the scores while in this scenario:

```
number of validation samples intersection over the union evaulated on 542
average intersection over union for background is 0.9956198659485371
average intersection over union for other people is 0.3554917669214411
average intersection over union for the hero is 0.8875822833070824
number true positives: 539, number false positives: 0, number false negatives: 0
```

We can see that the average IoU for the hero is about 89% so the model does a pretty good job of identifying the hero from up close.

![alt text][image3]

The next scenario is with the hero not in view of the drone. We can see that the model occasionally falsely identifies the hero. This can be seen by the blue patch in the predicted image below and by the large number of false positives in the scores.

```
number of validation samples intersection over the union evaulated on 270
average intersection over union for background is 0.9848841174787628
average intersection over union for other people is 0.6913552194187074
average intersection over union for the hero is 0.0
number true positives: 0, number false positives: 59, number false negatives: 0
```

![alt text][image4]

The final scenario is with the taget in view but far away. We can see by the image that the model does an extremely poor job of identifying the hero at a distance. This is made clear by the low IoU score for the hero of about 25%. This has a drastic effect on the performance of the drone in "follow me" mode since it will struggle to find the hero unless the hero is close by. This will significantly increase the target acquisition time. To best way to combat this is to capture more data with the hero at a distance which will allow the network to better kearn for this scenario.

```
number of validation samples intersection over the union evaulated on 322
average intersection over union for background is 0.9963200661728971
average intersection over union for other people is 0.4459581144813456
average intersection over union for the hero is 0.24758440845779545
number true positives: 138, number false positives: 3, number false negatives: 163
```

![alt text][image5]

The final jupyter notebook implementation can be found [here](https://github.com/thassan743/RoboND-DeepLearning-Project/blob/master/code/model_training.ipynb).

A saved html version of the notebook can be found [here](https://github.com/thassan743/RoboND-DeepLearning-Project/blob/master/model_training.html). All the results and images shown in this writeup can be found in the saved html notebook.

The weights for the model can be found [here](https://github.com/thassan743/RoboND-DeepLearning-Project/blob/master/data/weights/model_weights). This weights file can be used to test the model with the Udacity Follow Me Simulator found [here](https://github.com/udacity/RoboND-DeepLearning-Project/releases/tag/v1.2.2).

#### Future Enhancements

There is quite a bit of work that can be done to improve the performance of the network. The first thing I would do is gather a lot more data. Data can be collected to target the specific shortfalls of the existing model, most notably the identification of the hero from a distance. If a significant amount of data is gathered, most of which has the hero present in the image at varying distances, I think this will have a significant effect on the models performance. Additionally, one could use data augmentation techniques to increase the dataset size. A common one which I have seen being used is to flip each of the training images, thereby effectively doubling the size of the dataset without much work. Other techniques include applying varying degrees of rotation or colour distortions to the images. In all cases, the hyper-parameters would need to be adjusted to account for the extra data, most importantly the `batch_size` and `steps_per_epoch`.

The size and depth of the model could also be increased, however, this would add more complexity and increase training time. While this may yield a better score, the current network architecture performs reasonably well already, and the gains may not be worth the extra effort.

The default code in the notebook uses an image size of 160x160, however the original training images are of size 256x256. We are therefore throwing away a significant amount of image data. Using the full image may increase the performance of the network at the expense of some training time, however this has not been tested as yet.

Finally, changes to the Keras model, such as using a different optimizer, may also offer some performance improvements.

As an additional consideration, as mentioned in the rubric, how well would the current model and data work for following another object such as a dog or a car? While the model architecture may transfer well to other objects, the trained network certainly would not. The network was trained on data containing people and some other objects such as buildings and trees, but was trained to identify a specific person. Therefore, it would not work at all trying to identify any other object. However, if an equivalent dataset were captured of the new object and the existing model retrained on the new data, I do believe the network would perform reasonably well, as long as the data used was adequate.