# CNN Notes

- Conv layer filters are same size as image. The only time they are smaller is when the filter has an uneven padding and it ignores edge pixels. In that case we lose one/two pixels from the edges.
- We can use multiple filters, each having different weights. This increases the depth of the image while keeping the size same.
- Each filter focuses on a different feature of image (edges, curves, etc.)
- Conv layers are the same size as input image but have higher depth (no. of channels/filters). This means they have many more hyperparameters that do not contain much information.
- Maxpool layers are used to enhance the pixels with more information and discard the ones with low information.
- This reduces the size of the image while keeping the depth same.
- A Maxpool layer wih stride=X reduces the size of the image by a factor of X.
- Multiple pairs of Conv and Maxpool layers continously reduce the size of the image while increasing it's depth.
