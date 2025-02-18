# Models 

This directory contains models used in the project. The resnet and Deeplabv3 are modified from the torchvision 
implementations. The Resnet is modified to include dropout and the Deeplabv3 is modified to use the Resnet Backbone with
dropout, have multiple heads or output the intermediate feature vector. 