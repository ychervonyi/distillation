## Knowledge distillation experiments

### How does it work?

It seems like it works as follows:
1) Train teacher network (usually big and slow model, such as CNN) on a training dataset. Let number of classes be `N`.
2) Select a subset of training examples, transfer dataset, (or use the full training dataset) and run it throught the teacher model. Save its logits (outputs before softmax) for each example, `logits_t`. Length of logits vector is `dim(logits_t) = N`, subscript `_t` means teacher.
3) Modify transfer dataset labels such that `y_d = [y, logits_t]`, `_d` stands for distilled.
4) Define a student model (usually small and fast model, such as MLP). Number of outputs should be the same as the number of classes for a big model, `dim(logits_s)=N`, subscript `_s` means student.
5) Modify student model by adding one more layer, which serves as input thourh which logits of teacher model will be fed. Now the output of student model is `output_d=[softmax(logits_1), softmax(logits_2/T)]`, where `T` is a free parameter called temperature. Note that `dim(output_d)=2N`. `logits_1` is original logits of the student model, `logits_2` is a new input.
6) Define modified loss function as `L_d = lambda * l(y_true, y_pred) + l(y_soft, y_pred_soft)`.
7) Train distilled model on the modified transfer dataset.
8) Predictions made by the student model are extracted as the first half of its outputs.

### How to run the code

Dependencies: Keras, Tensorflow, Numpy

1) Train teacher model. 

CNN:

```python train.py --file data/matlab/emnist-letters.mat --model cnn```

or perceptron:

```python train.py --file data/matlab/emnist-letters.mat --model mlp```

2) Train student network with knowledge distillation:

```python train.py --file data/matlab/emnist-letters.mat --model student --teacher bin/10cnn_32_128_1model.h5```

### Results
[EMNIST-letters](https://www.nist.gov/itl/iad/image-group/emnist-dataset) dataset was used for experiments (26 classes of hand-written letters of english alphabet)

As a teacher network a simple cnn with `3378970` parameters (2 conv layers with 64 and 128 filters each, 1024 neurons on fully-connected layer) was trained for 26 epochs and was early stopped on plateau. Its validation accuracy was _94.4%_

As a student network a 1-layer perceptron with 512 hidden units and `415258` total parameters was used (8 times smaller than teacher network). First it was trained alone for 50 epochs, val acc was _91.6%_.

Knowledge distillation approach was used with different combinations of `temperature` and `lambda` parameters. Best performance was achieved with `temp=10, lambda=0.5`. Student network trained that way for 50 epochs got val acc of _92.2%_. 

So, the accuracy increase is less than 1% comparing to classicaly trained perceptron. But still we got some improvement. Actually all reports that people did, show similar results on different tasks: 1-2% quality increase. So we may say that reported results were reproduced on emnist-letters dataset. 

[Knowledge distillation](https://arxiv.org/abs/1503.02531) parameters (temperature and lambda) must be tuned for each specific task. To get better accuracy gain additional similar techniques may be tested, e.g. [deep mutual leraning](https://arxiv.org/abs/1706.00384) or [fitnets](https://arxiv.org/abs/1412.6550). 

