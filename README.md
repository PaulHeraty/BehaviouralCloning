HOW I APPROACHED THIS PROBLEM

I started off using a three layer CNN followed by a 2 layer fully connected network. For a long time, I was not having much luck with this. I read the paper that Nvidia have on the subject, and I modified my network to look like theirs. However, the best I could get was a constant value coming back from my model, regardless of the input image.

I started looking into the steering data that I was passing to the model generator. At the time I was using a keybaord for input, and the data was discontinuous, where there would be large steering angles followed by zeros for any given turn. I figured that this would confuse the model in training, as the images would look very similar, but the output angles were very different. I then tried a number of smoothing strategies, but could not get anything to work.  So I went off and purchased an analog joystick and tried to use this. The angle data looked a lot smoother, and my model started to produce different output angles for a range of images that I tested. 

The problem I was having now was that training was taking a long time. I was running on a Haswell quad-core CPU, and training could take anywhere from 5 to 12 hours, depending on the number of epochs run. This was proving very frustrating, and I needed to do a lot of 'what-if' work, and the turn around time was too long. I applied for a GPU VM with AWS, but this was ultimately denied by Amazon. In the meantime, I rebuilt my gaming laptop to dual-boot Linux. It had a GTX980M Nvidia card. I was now seeing ~20x speedup in training, so I could now do a lot more testing.

A key learning for me was to get a model that was working a little, and then fine tune that instead of training again from scratch. For example, I had a model that drove as far as the first bridge and then crashed. Initially, I added more training data and re-run full training again. With this approach I was getting 'random' results; sometimes better, sometimes much worse where it would not even get to the bridge. So I added a mode into my code called fine_tune_mode. In fine_tune_mode, I would lower the learning rate significantly, and also load an existing 'good' model and it's weights, and then tune it with new driving data. So my overall process looks like:

1. Create a new model with new training data
2. Repeat step 1 until I have a model that can drive a little part of the track
3. Save this 'good' model (json and h5), and also save it's IMG directory and .csv file
4. Gather some new training data for the section of the track that the model is having issues with
5. Run model.py in fine_tune_mode, where we load the 'good' model & weights, and train with a lower learning rate using the new data
6. Test this new model. If it's better than before, then repeat steps 3-6 until we can drive the full track
7. If the model is worse that before then discard the new training data and capture new data

Using this successive refinement approach, I was able to get the model driving the full track.

DESCRIPTION OF MY NETWORK

I'm using a 5 layer CNN followed by a 3 layer fully connected network. I use a lambda layer at the start to regularize the input image data. The first three convolution layers use 5x5 filters and the last two use 3x3 filters. I do not use any MaxPooling layers, but the convolution layers use striding instead to reduce dimensionality. The first three conv layers uses a 2x2 stride and the final two layers use a 1x1 stride. The depth of the filters increase, goinf from 24 in the first layer to 64 in the final CNN layer. I use dropout layers in between the CNN layers to help prevent over-fitting. Finally, each conv layer is followed by a ReLu activation layer.

There is a flatten layer in between the final convolution layer and the first dense layer in order to present a 2D array as input. The three dense layers are of widths 100, 50 and 1. There are ReLu activation layers in between layers 1 & 2, and 2 & 3. There is no softmax layer on the output, as this is a regression task (not a classification).

I am not using any weight regularization at this time.

HOW I CHOSE DATA TO INDUCE DESIRED BEHAVIOUR

I have already explained the process for tuning the model above. I started off by driving the track slowly a couple of times, generating around 40k images using to 50Hz simulator. I found that the 50Hz simulator was much better than the 10Hz one, as it generated image samples more frequently and thus the steering angles were smoother. 

Once I trained a model using this intial dataset, I looked at where the model was having issues on the track. If it was having trouble at a particular spot, I would then generate some new data by driving at that spot. I'd also add some recovery data, putting the car in 'bad' positions, turning on recording, and then driving the car back to a good spot on the track. As explained above, I would then put model.py in fine_tune_mode and update the existing model & weigths with this new data. I often had to play with the learning rate in fine_tune_mode so that it would not break the existing good behaviour in the model. There was a lot of trial and error here.

HOW I TRAINED MY MODEL

I think I have explained this sufficiently above.

THINGS THAT I LEARNED

There are about 10 things that I learned in this excercise.

1. How to use Python generators in Keras. This was critical as I was running out of memory on my laptop just trying to read in all the image data. Using generators allows me to only read in what I need at any point in time. Very useful.
2. Use a GPU. This should almost be a pre-requisite. It is too frustrating waiting for hours for results on CPU. I must have run training 100 times over the past 3 weeks and it was driving me crazy.
3. Use an analog joystick. This also should be a pre-requisite. I'm not sure if its even possible to train with keyboard input. Garbage in, garbage out.
4. Use successive refinement of a model. This really saves time and ensures that you converge ona solution faster.
5. Use the 50Hz simulator. This generates much smoother driving angle data. Should be the default.
6. You need at least 40k samples to get a useful initial model. Anything less was not producing anything good for me.
7. Copy the Nvidia pipeline. It works :)
8. Re-size the input image. I was able to size the image down by 2, reducing the number of pixel by 4. This really helped speed up model training and did not seem to impact the accuracy.
9. I made use of the left and right camera angles also, where I modified the steering angles slightly in these cases. This helped up the number of test cases, and these help cases where the car is off center and teaches it to steer back to the middle.
10. Around 5 epochs seems to be enough training. Any more does not reduce the mse much if at all.

