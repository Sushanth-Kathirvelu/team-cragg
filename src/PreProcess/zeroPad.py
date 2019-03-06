import cv2
import numpy as np

class zeroPad:
"""
this function zero pads the image into the given dimensions
by padding the image with zeros until it fits the given size.
"""
    def zeroPad ( old , target_dim=[640,640] ):
            """
            :old: the image to zero pad
            :target_dim:  the target size
            :return: returns the resized image
            """

            dims = old.shape
            if dims[0] != target_dim[0] or dims[1] != target_dim[1]:
                """
                if the image is not in the desired format change it to the desired size
                """
                if (len(dims) == 2):

                    temp_mask = np.zeros((target_dim[0], target_dim[1]))
                else:
                    temp_mask = np.zeros((target_dim[0], target_dim[1],3))

                temp_mask[:dims[0], :dims[1]] = source

                return temp_mask

            else:
                # if the image is already in the desired size, return it.
                return old
