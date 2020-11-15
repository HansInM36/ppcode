#!/bin/bash

# this script is for modifying the size of massive photoes in batch and generate a gif file

caseDir=/home/nx/OpenFOAM/project/w2f/waveType/1StokesFirst

cd $caseDir/animation/

# list all the pictures in picList
ls ./ > picList

# number of the pictures
picNum=`wc -l picList | awk '{print $1}'`;

# loop over the pictures and cut them into appropriate size
for ((index=1; index<$picNum; index++));
do
    # get the name of the current picture file
    picName=`sed -n "$[$index+1]p" picList` # must use double-quote
    echo "processing picture $index ..."
    convert $picName -crop 1536x200+0+340 mdf.$picName
    rm $picName
done

rm picList

# generate gif file
convert *.png anm.gif

cd $caseDir
