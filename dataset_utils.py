from __future__ import division
from __future__ import print_function

import h5py
import numpy as np
from scipy.io import loadmat

import os
#from PIL import Image
#import numpy as np
#from torchvision import transforms as T
#from datafolder.reid_dataset import import_MarketDuke_nodistractors
from datafolder.reid_dataset import import_Market1501Attribute_binary
#from datafolder.reid_dataset import import_DukeMTMCAttribute_binary

class DatasetLoader:
    """ Dataset loader class that loads feature matrices from given paths and
        create shuffled batch for training, unshuffled batch for evaluation.
    """
    def __init__(self, im_feat_path, sent_feat_path, split='train'):
        print('Loading image features from', im_feat_path)
        data_im = loadmat(im_feat_path)
        im_feats = np.array(data_im['img_feat']).astype(np.float32)
        print('Loaded image feature shape:', im_feats.shape)
        print('Loading sentence features from', sent_feat_path)
        #data_sent = h5py.File(sent_feat_path)
        # WARNING: Tanspose is applied if and only if the feature is stored as
        # a column in the original matrix.
        #sent_feats = np.array(data_sent['text_features']).astype(np.float32).transpose()
        #print('Loaded sentence feature shape:', sent_feats.shape)

        #train ,query, gallery = import_MarketDuke_nodistractors('/home/litongxin/Market-1501')
        train_attr, test_attr, self.label = import_Market1501Attribute_binary('/home/litongxin')

        self.split = split
        self.im_feat_shape = im_feats.shape
        #print(train_attr.values().shape)
        self.attr_feat_shape = np.array(list(train_attr.values())).shape
        self.img_inds = list(range(len(im_feats))) # we will shuffle this every epoch for training
        self.im_feats = im_feats
        self.attr_feats = train_attr
        #print(self.attr_feats)
        print(self.attr_feat_shape)
        #self.sent_feats = sent_feats
        # Assume the number of sentence per image is a constant.
        #self.sent_im_ratio = len(sent_feats) // len(im_feats)

    def shuffle_inds(self):
        '''
        shuffle the indices in training (run this once per epoch)
        nop for testing and validation
        '''
        if self.split == 'train':
            np.random.shuffle(self.img_inds)
            #np.random.shuffle(self.im_inds)

    def sample_index(self, index):
        im_id=self.im_feats['img_id'][index]
        start_ind=index
        end_ind=index
        while(self.im_feats['img_id'][start_ind]==im_id):
            start_ind=start_ind-1
        start_ind=start_ind+1
        while (self.im_feats['img_id'][end_ind] == im_id):
            end_ind = end_ind + 1
        end_ind = end_ind - 1
        return start_ind, end_ind

    def sample_items(self, sample_inds, sample_size):
        '''
        for each index, return the  relevant image and sentence features
        sample_inds: a list of sent indices
        sample_size: number of neighbor sentences to sample per index.
        '''
        attr_feat_b=[]
        for i in sample_inds:
            attr_feat_b.append(self.attr_feats[self.im_feats['img_id'][i]])
        #im_feats_b = self.im_feats[[i // self.sent_im_ratio for i in sample_inds],:]
        im_feats_b = []
        for ind in sample_inds:
            # ind is an index for image
            start_ind,end_ind = sample_index(ind)
            #start_ind = ind - ind % self.sent_im_ratio
            #end_ind = start_ind + self.sent_im_ratio
            sample_index = np.random.choice(
                    [i for i in range(start_ind, end_ind) if i != ind],
                    sample_size - 1, replace=False)
            sample_index = sorted(np.append(sample_index, ind))
            im_feats_b.append(self.im_feats[sample_index])
        im_feats_b = np.concatenate(im_feats_b, axis=0)
        return im_feats_b, attr_feat_b

    def get_batch(self, batch_index, batch_size, sample_size):
        start_ind = batch_index * batch_size
        end_ind = start_ind + batch_size
        #if self.split == 'train':
        sample_inds = self.img_inds[start_ind : end_ind]
        #else:
            # Since sent_inds are not shuffled, every self.sent_im_ratio sents
            # belong to one image. Sample each pair only once.
            #sample_inds = self.img_inds[start_ind * self.sent_im_ratio : \
            #                end_ind * self.sent_im_ratio : self.sent_im_ratio]
        (im_feats, attr_feats) = self.sample_items(sample_inds, sample_size)
        # Each row of the labels is the label for one sentence,
        # with corresponding image index sent to True.
        labels = np.repeat(np.eye(batch_size, dtype=bool), sample_size, axis=0)
        return(im_feats, attr_feats, labels)
