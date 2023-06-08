# --------------------------------------------------------
# Ke Yan,
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory (CADLab)
# National Institutes of Health Clinical Center,
# Apr 2019.
# This file contains codes of the network structure of LesaNet.
# --------------------------------------------------------

import torch.nn as nn
import torch
from roi_pooling.modules.roi_pool import _RoIPooling
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import Parameter

from utils import logger
from config import config


class VGG16bn(nn.Module):
    """VGG-16 multi-scale feature with batch normalization"""

    def __init__(self, num_cls, pretrained_weights=True):
        super(VGG16bn, self).__init__()
        self.num_cls = num_cls
        self._make_conv_layers(batch_norm=True)                                                             #creates a convolutional layer with batch normalization
        self._make_linear_layers(num_cls, roipool=5, fc=256, emb=config.EMBEDDING_DIM, norm=True)           #creates linear layers for classification and embedding
        if pretrained_weights:
            self._load_pretrained_weights()                                                                 #loads pretrained weights
        self._initialize_weights()

    def _load_pretrained_weights(self):
        pretr = models.vgg16_bn(pretrained=True)                                                            #loads pretrained weights from VGG16 with batch normalization
        pretrained_dict = pretr.state_dict().items()
        pretrained_dict = {k: v for k, v in pretrained_dict}
        model_dict = self.state_dict()
        pretr_keys = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]                                      #defines corresponding keys(layers) for loading the weights
        load_dict = {}
        for i in range(len(pretr_keys)):
            load_dict[self.conv_layer_names[i]+'.weight'] = pretrained_dict['features.%d.weight' % pretr_keys[i]]                   #extract and map weights to the corresponding layers
            load_dict[self.conv_layer_names[i]+'.bias'] = pretrained_dict['features.%d.bias' % pretr_keys[i]]
            load_dict[self.bn_layer_names[i]+'.weight'] = pretrained_dict['features.%d.weight' % (pretr_keys[i]+1)]
            load_dict[self.bn_layer_names[i]+'.bias'] = pretrained_dict['features.%d.bias' % (pretr_keys[i]+1)]
            load_dict[self.bn_layer_names[i]+'.running_mean'] = pretrained_dict['features.%d.running_mean' % (pretr_keys[i]+1)]
            load_dict[self.bn_layer_names[i]+'.running_var'] = pretrained_dict['features.%d.running_var' % (pretr_keys[i]+1)]

        model_dict.update(load_dict)                                                                                                #updates the dictionary with the new model data

    def _make_conv_layers(self, batch_norm=False):
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]                                   #defines the configuration of the convolutional layers
        block_idx = 1                                                                                                               #index for the blocks and layers
        layer_idx = 1
        in_channels = config.NUM_SLICES                                                                                             #number of input channels
        self.conv_layer_names = []                                                                                                  #creates lists to store the names of the convolutional layers
        self.bn_layer_names = []                                                                                                    #creates lists to store batch normalization names
        for v in cfg:
            if v == 'M':                                                                                                            #max pooling layer
                layer = nn.MaxPool2d(kernel_size=2, stride=2)
                name = 'pool%d' % block_idx
                exec('self.%s = layer' % name)                                                                                     
                block_idx += 1                                                                                                      #increments the block and layer
                layer_idx = 1
            else:
                layer = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                name = 'conv%d_%d' % (block_idx, layer_idx)
                self.conv_layer_names.append(name)                                                                                 #store the name of the convolutional layer
                exec('self.%s = layer' % name)                                                                                     #create a na,ed attribute with the layer
                if batch_norm:
                    layer = nn.BatchNorm2d(v)                                                                                      #batch normalization layer
                    name = 'bn%d_%d' % (block_idx, layer_idx)
                    exec('self.%s = layer' % name)
                    self.bn_layer_names.append(name)
                layer = nn.ReLU(inplace=True)                                                                                      #reLU activation function
                name = 'relu%d_%d' % (block_idx, layer_idx)                                                                        
                exec('self.%s = layer' % name)

                in_channels = v                                                                                                     #update the number of input channels for the next layer
                layer_idx += 1                                                                                                      #move to the next layer

    def _make_linear_layers(self, num_cls, roipool=5, fc=512, emb=1024, norm=True):
        lin_in = roipool * roipool
        self.roipool1 = _RoIPooling(roipool, roipool, 1)                    #first ROI pooling layer
        self.fc_pool1 = nn.Linear(lin_in * 64, fc)

        self.roipool2 = _RoIPooling(roipool, roipool, .5)                   #second ROI pooling layer
        self.fc_pool2 = nn.Linear(lin_in * 128, fc)

        self.roipool3 = _RoIPooling(roipool, roipool, .25)                  #third ROI pooling layer
        self.fc_pool3 = nn.Linear(lin_in * 256, fc)

        self.roipool4 = _RoIPooling(roipool, roipool, .125)                 #fourth ROI pooling layer
        self.fc_pool4 = nn.Linear(lin_in * 512, fc)

        self.roipool5 = _RoIPooling(roipool, roipool, .0625)                #fifth ROI pooling layer
        self.fc_pool5 = nn.Linear(lin_in * 512, fc)

        self.fc_emb = nn.Linear(fc*5, emb)                                  #fully connected layer for embedding
        self.class_scores1 = nn.Linear(fc * 5, num_cls)                     #fully connected layer for class scores
        self.class_scores2 = nn.Linear(num_cls, num_cls)                    #fully connected layer for score propogation

    def forward(self, inputs):
        x, out_box, center_box = inputs
        bs = x.size(0)
        idxs = torch.arange(bs).unsqueeze(1).to(torch.float).cuda()
        out_box = torch.cat((idxs, out_box), 1)
        center_box = torch.cat((idxs, center_box), 1)

        x = self.relu1_1(self.bn1_1(self.conv1_1(x)))                       #forward pass through convolutional layers and pooling layers   
        x = self.relu1_2(self.bn1_2(self.conv1_2(x)))
        pool1 = self.pool1(x)
        roipool1 = self.roipool1(x, center_box)
        roipool1 = roipool1.view(bs, -1)
        fc_pool1 = self.fc_pool1(roipool1)

        x = self.relu2_1(self.bn2_1(self.conv2_1(pool1)))                   #forward pass for the second set of layers
        x = self.relu2_2(self.bn2_2(self.conv2_2(x)))
        pool2 = self.pool2(x)
        roipool2 = self.roipool2(x, center_box)
        roipool2 = roipool2.view(bs, -1)
        fc_pool2 = self.fc_pool2(roipool2)

        x = self.relu3_1(self.bn3_1(self.conv3_1(pool2)))                   #forward pass for the third set of layers
        x = self.relu3_2(self.bn3_2(self.conv3_2(x)))
        x = self.relu3_3(self.bn3_3(self.conv3_3(x)))
        pool3 = self.pool3(x)
        roipool3 = self.roipool3(x, center_box)
        roipool3 = roipool3.view(bs, -1)
        fc_pool3 = self.fc_pool3(roipool3)

        x = self.relu4_1(self.bn4_1(self.conv4_1(pool3)))                   #forward pass for the fourth set of layers
        x = self.relu4_2(self.bn4_2(self.conv4_2(x)))
        x = self.relu4_3(self.bn4_3(self.conv4_3(x)))
        pool4 = self.pool4(x)
        roipool4 = self.roipool4(x, out_box)
        roipool4 = roipool4.view(bs, -1)
        fc_pool4 = self.fc_pool4(roipool4)

        x = self.relu5_1(self.bn5_1(self.conv5_1(pool4)))                   #forward pass for the fifth set of layers
        x = self.relu5_2(self.bn5_2(self.conv5_2(x)))
        x = self.relu5_3(self.bn5_3(self.conv5_3(x)))
        roipool5 = self.roipool5(x, out_box)
        roipool5 = roipool5.view(bs, -1)
        fc_pool5 = self.fc_pool5(roipool5)

        fc_cat = torch.cat((fc_pool1, fc_pool2, fc_pool3, fc_pool4, fc_pool5), dim=1)           #concatenated the 5 pooling layers
        emb = self.fc_emb(fc_cat)                                                               #passes the concatenated features through a fully connnected layer for embedding
        emb = F.normalize(emb, p=2, dim=1)                                                      #normalizes the embedding vectors

        out = {}
        out['emb'] = emb                                                                        #stores the normalized embedding vectors in the output dictionary

        class_score1 = self.class_scores1(fc_cat)                                               #passes concatenated features through a fully connected layer to obtain class scores
        class_prob1 = torch.sigmoid(class_score1)                                               #applies sigmoid activation function to the class scores to obtain class probabilities
        out['class_score1'] = class_score1                                                      #stores the class scores and probabilities in the dictionary
        out['class_prob1'] = class_prob1
        if config.SCORE_PROPAGATION:
            class_score2 = self.class_scores2(class_score1)                                     #passes the class scores through another fully connected layer to obtain class scores
            class_prob2 = torch.sigmoid(class_score2)                                           #applies sigmoid activation function to the propogated class scores to obtain class probabilities
            out['class_score2'] = class_score2                                                  #stores propogates class scores and probabilities to the dictionary
            out['class_prob2'] = class_prob2

        return out

    def _initialize_weights(self):          
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.weight.shape == (self.num_cls, self.num_cls):
                m.weight.data[:] = torch.eye(self.num_cls)                                      #initializes the weight matrix of specific linear layers
                m.bias.data.zero_()                                                             #initializes the bias vector to zere

#summary:
#The network is first initialized and the weights are loaded and put with the corresponding layers of the network
#the make_conv_layers function defines the convolutional layers with batch normalization
#The make_linear_layers function defines the ROI pooling layers and the corresponding fully connected layers for feature extraction
#It also defines the fully connected layer for embedding and for class scores
#The forward function does the forward pass of the network, computes the embedding vectors and class scores, applies activatin functions, and stores the results in the output dictionary
#The initialize_weights function initializes the weights of linear layers
