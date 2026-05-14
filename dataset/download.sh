#!/bin/bash

DISKMOUNTDIR=/data/yschoi/dataset/deephashing

mkdir ./dataset/ImageNet100
mkdir ./dataset/NUSWIDE
mkdir ./dataset/MSCOCO

# mkdir ${DISKMOUNTDIR}/CIFAR10
mkdir -p ${DISKMOUNTDIR}/Flickr25k/images
mkdir -p ${DISKMOUNTDIR}/ImageNet100/images
mkdir -p ${DISKMOUNTDIR}/NUSWIDE/images
mkdir -p ${DISKMOUNTDIR}/MSCOCO/images

# mkdir ${DISKMOUNTDIR}/Flickr25k/comp
mkdir ${DISKMOUNTDIR}/ImageNet100/comp
mkdir ${DISKMOUNTDIR}/NUSWIDE/comp
mkdir ${DISKMOUNTDIR}/MSCOCO/comp

# ln -s ${DISKMOUNTDIR}/CIFAR10 ./dataset
ln -s ${DISKMOUNTDIR}/Flickr25k/images ./dataset/Flickr25k
ln -s ${DISKMOUNTDIR}/ImageNet100/images ./dataset/ImageNet100
ln -s ${DISKMOUNTDIR}/NUSWIDE/images ./dataset/NUSWIDE/images
ln -s ${DISKMOUNTDIR}/MSCOCO/images ./dataset/MSCOCO/images

cd ${DISKMOUNTDIR}

# ## CIFAR10 ###########################################################################################
# # torchvision에서 알아서 다운로드 됨

# ## Flick25k ###########################################################################################
# wget http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k.zip -p Flickr25k/comp

# # [PATH]
# # Flickr25k/comp/mirflickr25k.zip

# # unzip Flickr25k/comp/mirflickr25k.zip -d Flickr25k/
# # mv Flickr25k/mirflickr/* Flickr25k/images
# # rm -r Flickr25k/mirflickr

# unzip ${DISKMOUNTDIR}/data/press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k.zip -d ${DISKMOUNTDIR}/Flickr25k/
# mv ${DISKMOUNTDIR}/Flickr25k/* Flickr25k/images
# rm -r ${DISKMOUNTDIR}/data/press.liacs.nl
# ## ImageNet100 ###########################################################################################
# https://image-net.org/download-images.php

# wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar -p ImageNet100/comp
# wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar -p ImageNet100/comp

# # [PATH]
# # ImageNet100/cmp/ILSVRC2012_img_train.tar
# # ImageNet100/cmp/ILSVRC2012_img_val.tar

# mkdir ${DISKMOUNTDIR}/ImageNet100/tmp
# mkdir ${DISKMOUNTDIR}/ImageNet100/images/train
# mkdir ${DISKMOUNTDIR}/ImageNet100/images/val
# tar xvf ImageNet100/comp/ILSVRC2012_img_train.tar --directory ImageNet100/tmp &
# tar xvf ImageNet100/comp/ILSVRC2012_img_val.tar --directory ImageNet100/images/val &
# cd ImageNet100/tmp/
# for f in *.tar; do tar -xvf "$f" --directory /backup/yschoi/data/ImageNet100/images/train ;done
# cd ../..
# rm -r ImageNet100/tmp


# ## MSCOCO ###########################################################################################
# # Chrome 브라우저 켜서 다운로드: https://drive.google.com/uc?id=0B7IzDz-4yH_HN0Y0SS00eERSUjQ

# # [PATH]
# # MSCOCO/comp/coco.tar.gz

# mv ${DISKMOUNTDIR}/coco.tar.gz ${DISKMOUNTDIR}/MSCOCO/comp/coco.tar.gz

# tar xvf MSCOCO/comp/coco.tar.gz --directory MSCOCO/comp
# unzip MSCOCO/comp/train2014.zip -d MSCOCO/images
# unzip MSCOCO/comp/val2014.zip -d MSCOCO/images

# ## NUSWIDE ###########################################################################################
# # Chrome 브라우저 켜서 다운로드: https://zdtnag7mmr.larksuite.com/file/boxusGL7TdZ1575dVRmgjgXq4dg

# # [PATH]
# # NUSWIDE/comp/Flickr.zip
# mv ${DISKMOUNTDIR}/Flicker.zip ${DISKMOUNTDIR}/NUSWIDE/comp/Flicker.zip

# unzip ${DISKMOUNTDIR}/NUSWIDE/comp/Flicker.zip -d NUSWIDE
# mv NUSWIDE/Flicker/* NUSWIDE/images
# rm -r NUSWIDE/Flicker/