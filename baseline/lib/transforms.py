
from torchvision import transforms
import numpy as np
import cv2

class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


default_transform = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

flip_randcrop_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((256,256)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


flip_rot_randcrop_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=(0,180)),
        transforms.Resize((300,300)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


centercrop_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


cibhash_transform = transforms.Compose([transforms.RandomResizedCrop(size = 224,scale=(0.5, 1.0)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p = 0.7),
                                            transforms.RandomGrayscale(p  = 0.2),
                                            GaussianBlur(3),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
                                            ])
    



TRAIN_TRANSFORM = { 'CIFAR10': default_transform, 
                    'Flickr25k': default_transform, 
                    'ImageNet100': default_transform, 
                    'MSCOCO': flip_randcrop_transform, 
                    'NUSWIDE': default_transform}
                    

TEST_TRANSFORM = { 'CIFAR10': default_transform, 
                    'Flickr25k': default_transform, 
                    'ImageNet100': default_transform, 
                    'MSCOCO': centercrop_transform, 
                    'NUSWIDE': centercrop_transform}

TRANSFORMS = {"default" : default_transform,
                 "Flip+RandCrop": flip_randcrop_transform,
                 "Flip+Rot+RandCrop": flip_rot_randcrop_transform,
                 "CenterCrop": centercrop_transform,
                 "CIB": cibhash_transform
                 }

def load_transform_by_dataset_name(dataset_name):
    return TRAIN_TRANSFORM[dataset_name], TEST_TRANSFORM[dataset_name]

def load_transform_by_transform_name(transform_name):
    return TRANSFORMS[transform_name]