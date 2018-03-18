# RefineDet_chainer
Benchmarking RefineDet[1] and other SSD (Single shot Detection) Network based on chainer

Including

- DSSD[2] 
- SSD with residual prediction module[2]
- ESSD[3]
- RefineDet[1]

Original DSSD is based on ResNet 101.
Since memory limitation, only tried on VGG.

# Benchmark result

|Model name |Base model |Input image size |mAP |mAP(paper)|
|-----------|-----------|-----------------|----|----------|
|SSD |VGG16 |300x300 |77.5|77.5|
|SSD Plus(Use Residual Prediction module) |VGG16 |300x300 |78.0 |NA|
|ESSD |VGG16 |300x300 |78.8 |79.4|
|RefineDet |VGG16 |320x320 |Now evaluating |80.0|

*: I set batchsize to 22 because of memory limitation. The original paper used 32.

*: Some training condition is different from paper.

*: ESSD original paper did 3 stages training (Only SSD, Only extensional module and whole network), but I did whole training only.

*: I may mistook unintensionally.

# Requirement

+ [Chainercv](https://github.com/chainer/chainercv) and its dependencies

# Usage

```
git clone https://github.com/fukatani/RefineDet_chainer
cd refinedet
python train.py --model refinedet320 --batchsize 22
```

# Other

Many implementation is referenced chainercv. Thanks!

# Reference


+ [Single-Shot Refinement Neural Network for Object Detection](https://arxiv.org/abs/1711.06897)

+ [DSSD : Deconvolutional Single Shot Detector](https://arxiv.org/abs/1701.06659)

+ [Extend the shallow part of Single Shot MultiBox Detector via Convolutional Neural Network](https://arxiv.org/abs/1801.05918)

