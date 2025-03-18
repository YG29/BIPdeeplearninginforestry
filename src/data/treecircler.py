import numpy as np
from skimage import measure, morphology

class TreeBoxMaker:
    '''Preprocess mask images to get bounding boxes'''
    def __init__(self, separation_method='binary_erosion', erosion_footprint=6):
        '''

        :param separation_method: method to separate touching trees (so far binary_erosion works best)
        :param erosion_footprint: parameter for binary_erosion
        '''
        self.erosion_footprint = erosion_footprint
        self.separation_method = separation_method


    def preprocess(self, mask):
        '''
        preprocess the mask image
        :param mask: input mask image
        :return: processed mask, and bounding boxes
        '''
        binary_mask = np.ndarray(mask)

        # separation method
        if self.separation_method == 'binary_erotion':
            separated_binary_mask = self.binary_erotion(binary_mask)
        else:
            separated_binary_mask = binary_mask

        # extract bounding box
        bboxes = self.bbox_extract(separated_binary_mask)

        return separated_binary_mask, bboxes

    def binary_erotion(self, binary_mask):
        separated_binary_mask = morphology.binary_erosion(
            binary_mask,
            footprint=np.ones((self.erosion_footprint, self.erosion_footprint))
        )
        return separated_binary_mask

    def bbox_extract(self, separated_binary_mask):
        labels = measure.label(separated_binary_mask, connectivity=2)
        region_props = measure.regionprops(labels)
        bboxes = []
        for region in region_props:
            bbox = region.bbox  # returns: (min_row, min_col, max_row, max_col)
            bboxes.append(bbox)
        return bboxes