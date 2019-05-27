---
published: true
---
## Torchvision package

PyTorch has `torchvision` package designed to prepare visual images for learning process.

The `torchvision` package in turn has additional subpackages.

    datasets 
    models 
    transforms 
    utils
    
The `torchvision.datasets` subpackage contains most important datasets. At the current moment these are:

    cifar
    cityscapes
    coco
    fakedata
    flickr
    folder
    lsun
    mnist
    omniglot
    phototour
    sbu
    semeion
    stl10
    svhn
    utils
    voc

The `torchvision.models` subpackage contains these models at the current moment:

    alexnet
    densenet
    inception
    resnet
    squeezenet
    vgg
    
The `torchvision.utils` help us save Tensors to a file. These tensors are of shape:

     BxCxHxW : number of mini batches, channels, height, width

and create grids of images.

But the most interesting sub-package today is the `torchvision.transforms` package.
This package has exactly two sub pakages `torchvision.transforms.functional` and `torchvision.transforms.transforms` that holds the classes behind the `torchvision.transforms.functional` methods.

The `torchvision.transforms.functional` package depends on `PIL.Image` functionality. Contains methods to detect the image type:

    _is_numpy_image
    _is_pil_image
    _is_tensor_image
    
Methods to adjust the image:
    
    adjust_brightness
    adjust_contrast
    adjust_gamma
    adjust_hue
    adjust_saturation
    
Methods to transform the image

    affine (keeps the center in place)
    center_crop    
    crop    
    five_crop
    hflip
    pad
    resize
    resized_crop
    rotate
    scale
    ten_crop
    vflip
    
Some handy methods to convert the image:

    to_grayscale
    to_pil_image
    to_tensor


And also the method to normalize the image.

    normalize
