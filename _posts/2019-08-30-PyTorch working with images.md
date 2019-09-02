---
published: true
layout: post
title: Working with images in Python, PyTorch
---

[Matplotlib](https://github.com/matplotlib/matplotlib) is one of the default choices when plotting images in Python and should always be considered first.

There are two interesting libraries you can read images and transform them. Both these work from Jupyter notebooks. These are PIL and OpenCV.

(I) PIL library that also has handy `display` method.
    
    %matplotlib inline
    import PIL    
    from PIL import Image
    import matplotlib.pyplot as plt

    img = PIL.Image.open("/data/image1234.JPEG")
    img.show()  # will open in external program
    display(img) # display on any frontend

    ToTensor = torchvision.transforms.ToTensor()
    FromTensor = torchvision.transforms.ToPILImage()
    t = FromTensor(ToTensor(img))
    plt.imshow(t) # matplotlib

The program above will show the following:

![IMG](/images/pimage1.png)

(II) OpenCV is another option to work with images in Python. (It also supports Videos) 

    %matplotlib inline
    import cv2
    import matplotlib.pyplot as plt

    img = "/data/image1234.JPEG"
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # color
    plt.imshow(img)
    cv2.imshow('image',img) # opens external program, may broke Jupyter session...

When printing images it is always right choice to use matplotlib.
Note that cv2 is made to return numpy arrays.

Another thing we are interesting are image transformations. It appears that cv2 is 3-5 times faster than PIL based on my previous checks.

