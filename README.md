# EECS442-project
Automatically add caption to given pictures is a basic problem in artiﬁcial 
intelligence which requires the cooperation of computer vision and natural 
language processing. There have been diverse approaches for this problem, here 
we are going to implement an encoder-decoder architecture. The encoder is made of 
a pre-trained ResNet and the decoder is mainly a word embedding layer and a LSTM. 
We will use perplexity as a metric to evaluate our caption.

## Usage

####1. Clone the repositories
```angular2
$ git clone https://github.com/pdollar/coco.git
$ cd coco/PythonAPI/
$ make
$ python setup.py build
$ python setup.py install
$ cd ../../
$ git clone https://github.com/zyihong/EECS442-project.git
$ cd EECS442-project
```

####2. Download the dataset
```angular2
$ pip install -r requirement.txt
$ chmod +x download.sh
$ ./download.sh
```

####3. Prepare data
```angular2
$ python prepare_data.py
```

####4. Train the model
```angular2
$ python train.py
```

####5. Auto-captioning the images
```angular2
$ python sample.py
```

<br>

## Notes
If you want to change the training parameters, 
you can go to `./train.py` and modify the parameters as you want.