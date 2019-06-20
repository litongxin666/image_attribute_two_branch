from __future__ import division
from __future__ import print_function

import h5py
import numpy as np
import scipy.io
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
        im_id = data_im['img_id']


        data_im_age=loadmat('/home/litongxin/Person-Attribute-Recognition-MarketDuke/img_age.mat')
        self.img_feats_age = np.array(data_im_age['img_feat']).astype(np.float32)
        self.img_id_age = data_im_age['img_id']
        self.img_inds_age = list(range(len(self.img_feats_age)))

        print('Loaded image feature shape:', im_feats.shape)
        print('Loading sentence features from', sent_feat_path)
        attr_id=loadmat("/home/litongxin/image_attribute_two_branch/attr_id.mat")
        print("shape",len(attr_id['attr_id']))

        attr_id_age=loadmat('/home/litongxin/Person-Attribute-Recognition-MarketDuke/datafolder/attr_age.mat')
        train_attr_age, test_attr_age, self.label = import_Market1501Attribute_binary('/home/litongxin')
        attr_id_not = []
        for j in train_attr_age.keys():
            if j not in attr_id_age['attr_age']:
                attr_id_not.append(j)
        # print("not",attr_id_not)
        for m in attr_id_not:
            train_attr_age.pop(m)
        self.train_attr_age=train_attr_age

        #print("attr_id",sorted(attr_id['attr_id']))
        #data_sent = h5py.File(sent_feat_path)
        # WARNING: Tanspose is applied if and only if the feature is stored as
        # a column in the original matrix.
        #sent_feats = np.array(data_sent['text_features']).astype(np.float32).transpose()
        #print('Loaded sentence feature shape:', sent_feats.shape)

        #train ,query, gallery = import_MarketDuke_nodistractors('/home/litongxin/Market-1501')
        train_attr, test_attr, self.label = import_Market1501Attribute_binary('/home/litongxin')
        attr_id_not=[]
        for j in train_attr.keys():
            if j not in attr_id['attr_id']:
                attr_id_not.append(j)
        #print("not",attr_id_not)
        for m in attr_id_not:
            train_attr.pop(m)
        #result={'attr_id':attr_id}
        #scipy.io.savemat('attr_id.mat', result)
        #attr_id=loadmat("/home/litongxin/image_attribute_two_branch/attr_id.mat")
        #print("attr_id",attr_id['attr_id'])
        self.split = split
        self.im_feat_shape = im_feats.shape
        #print(train_attr.values().shape)
        self.attr_feat_shape = np.array(list(train_attr.values())).shape
        #print("attr_shape",self.attr_feat_shape)
        self.attr_test_feat_shape = np.array(list(test_attr.values())).shape
        self.img_inds = list(range(len(im_feats))) # we will shuffle this every epoch for training
        self.im_feats = im_feats
        self.attr_feats = train_attr
        self.attr_test_feats = test_attr
        self.im_id = im_id
        #print(self.attr_feats)
        #print("test_attr_feat_shape",self.attr_test_feat_shape)
        #print("test_img_shape",self.im_feat_shape)
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
        im_id=self.im_id[index][0]
        start_ind=index
        end_ind=index
        while(self.im_id[start_ind][0]==im_id and start_ind>=0):
            start_ind=start_ind-1
        start_ind=start_ind+1
        while (self.im_id[end_ind][0] == im_id and end_ind<8835):
            end_ind = end_ind + 1
        end_ind = end_ind - 1
        return start_ind, end_ind

    def sample_items(self, sample_inds, sample_size):
        '''
        for each index, return the  relevant image and sentence features
        sample_inds: a list of sent indices
        sample_size: number of neighbor sentences to sample per index.
        '''
        #print("sample_index",sample_inds)
        attr_feat_b=[]
        for i in sample_inds:
            #print("attr_index",self.im_id)
            #print("attr",self.attr_feats[strt])
            #start_ind, end_ind = self.sample_index(i)
            attr_feat_b.append(self.attr_feats[self.im_id[i][0]])
            #attr_feat_b.append(self.attr_feats[self.im_feats['img_id'][i]])
        #im_feats_b = self.im_feats[[i // self.sent_im_ratio for i in sample_inds],:]
        #print("len_attr",len(attr_feat_b))
        im_feats_b = []
        for ind in sample_inds:
            # ind is an index for image
            start_ind,end_ind = self.sample_index(ind)
            if end_ind-start_ind==1:
                sample_index=np.append(start_ind,end_ind)
            else:
                sample_index = np.random.choice(
                    [i for i in range(start_ind, end_ind) if i != ind],
                    sample_size - 1, replace=False)
                #print("sample",sample_index)
                sample_index = sorted(np.append(sample_index, ind))
            im_feats_b.append(self.im_feats[sample_index])
            #else:
            #    sample_index=[]
            #    count=sample_size-(end_ind-start_ind+1)
            #    for i in range(start_ind,end_ind+1):
            #        sample_index.append(i)
            #    if count!=0:
            #        for j in range(count):
            #            sample_index.append(ind)
            #    im_feats_b.append(self.im_feats[sample_index])
            #print("sample",sample_index)
        #print("len_im_feat",len(im_feats_b))
        im_feats_b = np.concatenate(im_feats_b, axis=0)
        return (im_feats_b, attr_feat_b)

    def test_sample_index(self,index):
        im_id = self.im_id[index][0]
        start_ind = index
        end_ind = index
        while (self.im_id[end_ind][0] == im_id and end_ind < 13114):
            end_ind = end_ind + 1
        end_ind = end_ind - 1
        if end_ind-start_ind>=1:
            return start_ind, end_ind
        else:
            return -1,end_ind

    def test_sample_items(self,sample_inds,sample_size):
        attr_feat_b=[]
        im_feats_b=[]
        ind=0
        while ind<13114:
            start_ind,end_ind=self.test_sample_index(ind)
            #print(start_ind,end_ind)
            if start_ind!=-1:
                attr_feat_b.append(self.attr_test_feats[self.im_id[ind][0]])
                sample_index = np.random.choice(
                    [i for i in range(start_ind, end_ind) if i != ind],
                    sample_size - 1, replace=False)
                # print("sample",sample_index)
                sample_index = sorted(np.append(sample_index, ind))
                #print(sample_index)
                #print("*"*10)
                im_feats_b.append(self.im_feats[sample_index])
            ind = end_ind+1
        im_feats_b = np.concatenate(im_feats_b, axis=0)
        return (im_feats_b, attr_feat_b)


    def get_batch(self, batch_index, batch_size, sample_size):
        start_ind = batch_index * batch_size
        end_ind = start_ind + batch_size
        #print("start",start_ind)
        #print("end",end_ind)
        if self.split == 'train':
            sample_inds = self.img_inds[start_ind : end_ind]
            #print("len",len(sample_inds))
            (im_feats, attr_feats) = self.sample_items(sample_inds, sample_size)
            labels = np.repeat(np.eye(batch_size, dtype=bool), sample_size, axis=0)
        else:
            sample_inds = self.img_inds[start_ind : 13115]
            #(im_feats,attr_feats) = self.test_sample_items(sample_inds, sample_size)
            im_feats = self.im_feats
            attr_feats=[]
            attr_feats_temp = sorted(self.attr_test_feats)
            for i in attr_feats_temp:
                attr_feats.append(self.attr_test_feats[i])
            labels = np.repeat(np.eye(707, dtype=bool), sample_size, axis=0)
        # Each row of the labels is the label for one sentence,
        # with corresponding image index sent to True.
        #labels = np.repeat(np.eye(batch_size, dtype=bool), sample_size, axis=0)
        return(im_feats, attr_feats, labels)


    def get_batch_age(self, batch_index, batch_size, sample_size):
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        sample_inds = self.img_inds_age[start_index: end_index]
        #print("start",start_ind)
        #print("end",end_ind)
        if self.split == 'train':
            attr_feats = []
            im_feats=[]
            for index in sample_inds:
                for i in range(4):
                    self.train_attr_age[self.img_id_age[index]][i]=2.0
                attr_feats.append(self.train_attr_age[self.img_id_age[index]])
                #attr_feat_b.append(self.attr_feats[self.im_id[i][0]])
                im_id = self.img_id_age[index][0]
                start_ind = index
                end_ind = index
                while (self.img_id_age[start_ind][0] == im_id and start_ind >= 0):
                    start_ind = start_ind - 1
                start_ind = start_ind + 1
                while (self.img_id_age[end_ind][0] == im_id and end_ind < mougeshu):
                    end_ind = end_ind + 1
                end_ind = end_ind - 1
                sample_index = np.random.choice(
                    [i for i in range(start_ind, end_ind)],
                    sample_size - 1, replace=False)
                im_feats.append(self.im_feats[sample_index])
            im_feats = np.concatenate(im_feats, axis=0)
            labels = np.repeat(np.eye(batch_size, dtype=bool), sample_size, axis=0)
            return (im_feats, attr_feats, labels)
        else:
            sample_inds = self.img_inds[start_index : 13115]
            #(im_feats,attr_feats) = self.test_sample_items(sample_inds, sample_size)
            im_feats = self.im_feats
            attr_feats=[]
            attr_feats_temp = sorted(self.attr_test_feats)
            for i in attr_feats_temp:
                attr_feats.append(self.attr_test_feats[i])
            labels = np.repeat(np.eye(707, dtype=bool), sample_size, axis=0)
        # Each row of the labels is the label for one sentence,
        # with corresponding image index sent to True.
        #labels = np.repeat(np.eye(batch_size, dtype=bool), sample_size, axis=0)
        return(im_feats, attr_feats, labels)
