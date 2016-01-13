# About

This project generates new christmas-related images using the technique of generative adversarial networks ([GAN](http://papers.nips.cc/paper/5423-generative-adversarial-nets)). The used architectures mostly match the one described in the [DCGAN](http://arxiv.org/abs/1511.06434) paper.

# Results

Three datasets were used during the training: Images of baubles (the spheres on christmas trees), images of whole christmas trees (in rooms) and images of landscapes with snow. The datasets were created specifically for this project by gathering images from [flickr](http://www.flickr.com) using relevant search terms and manually cleaning up the results afterwards. All datasets were augmented to several times their original size. Some training runs were done in grayscale mode to decrease the difficulty of the problem. Exact architectures of each model can be viewed in `images/architectures.txt`.

## Baubles (2475 images)

This dataset turned out to be very hard to generate. The number of example images was low while the variance was (apparently) high, resulting in the network being unable to correctly learn a generalizing function. That was the case for both 64x64 and 32x32 images (in many different tested architectures). No architecture produced a significant amount of images that clearly resembled baubles. From the perspective of a human that seems odd, as the dataset didn't appear to be *that* difficult: Learn a greenish background texture, learn 1 to 10 different bauble styles, place 1 to 4 baubles randomly in the image - that should already be enough, but somehow no network managed to do it. I tried shallow architectures for D (and high dropout rates), so it was probably not related to D memorizing the dataset. Maybe these networks have problems with drawing perfect spheres. Or maybe it's related to how hard it sometimes is to spot the baubles in the images (e.g. glass baubles or greenish baubles). Or maybe the variance was really just too high.

**Training set (excerpt):**

![Example images from the training set](images/baubles_64x64_2_trainset.jpg?raw=true "Example images from the training set")

**Generated images (examples):**

![Generated images](images/baubles_64x64_2_best.jpg?raw=true "Generated images")

*Examples of generated images. You can see some indication that it starts to learn what baubles look like, but it's still rather far off.*

## Christmas Trees (2090 images)

This dataset worked pretty well. The variance of the main eye catcher (the tree) is rather low. The rooms in the background are mostly wrong/broken, but that's not easily noticeable from far away, only when zoomed in. Many trees are also partly broken - again, not easy to see when zoomed out. The network sometimes accidently generates two trees instead of one. At least the errors regarding the trees would probably go away with more example images.

**Training set (excerpt):**

![Example images from the training set](images/trees_trainset.jpg?raw=true "Example images from the training set")

**Generated images (examples):**

![Generated images](images/trees64_3_e1230_rnd256.jpg?raw=true "Generated images")

**Training video:**

[![Training progress video](images/christmas-trees-youtube.jpg?raw=true)](https://youtu.be/EOylC-JsLFE)

## Snowy Landscapes (10201 images)

This dataset had pretty high variance, but also contained more images than the other two datasets. The generated images are not great, but not completely horrible either. Tried a lot of architectures, including deep residual generators, but in the end fairly standard architectures produced the best results.

**Training set (excerpt, grayscale run):**

![Example images from the training set](images/snow_gray_trainset.jpg?raw=true "Example images from the training set")

**Generated images (examples, grayscale run):**

![Generated images](images/snow_64x96_2_e1380_rnd256.jpg?raw=true "Generated images")


**Training set (excerpt, color run):**

![Example images from the training set](images/snow_color_trainset.jpg?raw=true "Example images from the training set")

**Generated images (examples, color run):**

![Generated images](images/snow_64x96_rnd256.jpg?raw=true "Generated images")

# Usage

Requirements are:
* Torch
  * Required packages (most of them should be part of the default torch install, install missing ones with `luarocks install packageName`): `cudnn`, `nn`, `pl`, `paths`, `image`, `optim`, `cutorch`, `cunn`, `cudnn`, `dpnn`, `display`
* Python 2.7 (might work with newer versions, not tested)
  * scikit-image
  * scipy
  * numpy
* NVIDIA GPU with cudnn3 and 4GB or more memory

To generate the dataset:
* clone the repository and `cd` into it
* `cd dataset`
* `python download_images.py` - This downloads all required images at a rate of 1 image per second, which will take quite some time.
* `python preprocess_images.py` - This will augment and normalize (baubles and christmas trees to 64x64, landscapes to 64x96) all downloaded images.

To train a network:
* `~/.display/run.js &` - This will start `display`, which is used to plot results in the browser
* Open http://localhost:8000/ in your browser (`display` interface)
* Open a console in the repository directory and use any of the following commands:
  * `th train.lua --profile="baubles32"` - Train a network to generate images of baubles in resolution 32x32
  * `th train.lua --profile="baubles64"` - Train a network to generate images of baubles in resolution 64x64
  * `th train.lua --profile="trees32"` - Train a network to generate images of christmas trees in resolution 32x32
  * `th train.lua --profile="trees64"` - Train a network to generate images of christmas trees in resolution 64x64
  * `th train.lua --profile="snow32"` - Train a network to generate images of snowy landscapes in resolution 32x48
  * `th train.lua --profile="snow64"` - Train a network to generate images of snowy landscapes in resolution 64x96

You can watch how the results of the network improve in the opened browser window. The training will continue until you stop it manually. By default the network is saved every 30 epochs. It also saves images at every epoch in `logs/images`, `logs/images_good` and `logs/images_bad`. Those images can take up quite some space over time, so keep an eye on it.
You can continue a training run at a later time by adding `--network="logs/adversarial.net"` as a parameter.
You can sample images from a trained network using `th sample.lua --profile="baubles32"` (analogous for the profiles `baubles64`, `trees32`, ...).

# Lessons learned

* Even in the case of low variance datasets 2k images seem to not be enough to generate images very few errors. Aim at >5k.
* For high variance datasets even 10k seem to not be enough.
* Results downloaded from flickr by keyword can contain a lot of off-topic stuff. When searching for `christmas baubles` I got images of police cars, when searching for `snow landscape` I got images of leopards. While cleaning up I regularly had to remove around half of all images and then probably still had lots of bad ones remaining.
* Evade cleaning up results downloaded from flickr manually. Takes ages of time.
