"""
Potentially helpful utilities for preparing entries for the CS444 
Image classification project.

author: Nathan Sprague
version: 10/2013
"""

import numpy as np
import matplotlib.pyplot as plt

def data_to_img(data):
    """
    Convert a single row of the image data array so that it is
    formatted as an image.

    The resulting array will have the shape: h x w x 3 where h is the
    number of rows, w the number of columns, and 3 is the number of
    colors in the image-- 0 corresponds to red, 1 corresponds to green,
    and 2 corresponds to blue.

    For example, the following statement accesses the red
    value of the pixel at row 20 colum 13 of the image.
  
                         row  col color
                           \   \   \
                            v   v   v
                        img[20, 13, 0]
    
    Parameters: 
    data - A length 3072 numpy array representing a single image
    
    Returns:
    a 32x32x3 numpy image.

    """
    p = np.empty(data.size, data.dtype)
    p[0::3] = data[:data.size/3]
    p[1::3] = data[data.size/3:2 * data.size/3]
    p[2::3] = data[2* data.size/3:]
    return p.reshape(32, 32, 3)


def generate_submission(labels, file_name):
    """ 
    Generate a .csv file suitable for submission to the Kaggle
    competition.
    """
    out = [[i, labels[i]] for i in range(labels.size)]
    f = open(file_name, 'wb')
    f.write(b'ImageId,PredictedClass\n')
    np.savetxt(f, out, delimiter=',', fmt='%d,%d')

def show_results(data, labels):
    """
    This is a utility function that can be used to visualize which
    images end up in which class.
    
    Parameters:
    data - A set of images in the original data format
    labels - one 0,1 label for each image.

    """
    implot = None
    for i in range(data.shape[0]):
        if implot == None:
            implot = plt.imshow(data_to_img(data[i,:]), 
                                interpolation='none')
            txt = plt.text(5, 5, str(labels[i]),size='20', 
                           color=(0,1,0))
        else:
            implot.set_data(data_to_img(data[i,:])) 
        txt.set_text(str(labels[i]))
        fig = plt.gcf()
        fig.set_size_inches(2,2)
        plt.xticks(())
        plt.yticks(())
        plt.draw()
        plt.waitforbuttonpress()
