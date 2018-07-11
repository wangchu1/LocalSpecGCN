## Local Spectral Graph Convolution for Point Set Feature Learning
This repository contains source code of the paper [*"
Local Spectral Graph Convolution for Point Set Feature Learning."*](https://arxiv.org/abs/1803.05827) The paper is accepted to ECCV 2018. 

### System Pipeline
![LocalSpecGCN pipeline](http://www.cim.mcgill.ca/~chuwang/files/LocalSpecGCN/framework.png)

### Citation
Should the paper or the code in this repository be useful to your research, please consider citing:

        @article{wang2018local,
          title={Local Spectral Graph Convolution for Point Set Feature Learning},
          author={Wang, Chu and Samari, Babak and Siddiqi, Kaleem},
          journal={arXiv preprint arXiv:1803.05827},
          year={2018}
        }


### **Code Credit Clarification:**
The code of this project is implemented by <a href="http://www.cim.mcgill.ca/~chuwang/index.html">Chu Wang</a> 
and is built on top of <a href="https://github.com/charlesq34/pointnet2">PointNet++</a>. We are grateful to the authors of pointnet++ for sharing their code base. The pointnet framework itself has been modified considerably to support the spectral graph kernels as well as cluster pooling module. The core implementation of graph convolution layers, several handy Graph Signal Processing fuctions as well as the cluster pooling module can be found in utils/spec_graph_util.py. The layer wrapper function of spectral graph convolution layer stays in utils/pointnet_util.py, where the other original pointnet++ layers reside. 

### Installation and Setup
#### - Tensorflow, Python and CUDA
You need to install <a href="https://www.tensorflow.org/install/">TensorFlow</a>, with TF version preferrably being 1.2 or 1.3 and python verison being 2.7. I highly recommend using virtual environment for tensorflow installation. Python 3 should work but we conducted no test. This code repository is tested to work under Ubuntu 14.04 and 16.04 with CUDA version 8.0. We are not sure about whether our code will work under other system configurations. It is recommended that you run the experiments under this repository using a decent GPU. We expect the training of our models to coverge after 15-20 hours using a single Titan Xp.

#### - Compiling point set operation fuctions in tf_ops
You need to compile the tf fuctions for basic point set operations in folder tf_ops. Simply open a terminal, go to these folders and run "compile.sh" or "xxxx_compile.sh" by 
```
bash compile.sh
```
Of course, you need to set up your path variables in those compile.sh like scripts. For more details, please refer to compile instructions in <a href="https://github.com/charlesq34/pointnet2">PointNet++</a>. In short, once your path is set correctly, it is very straightforward to compile these handy functions. 

#### - Download the data for training/testing
We used the same data in this paper as pointnet++, for fair comparison purposes. Please go to data folder, follow instructions there and download the data. You are recommended to put unziped modelnet and shapenet folders in data/, but they can of course go somewhere else. Simply modify "DATA_DIR" variable to point to your data directory of choice, in our training scripts.

#### - You are ready to go!


### Running Experiments
#### - ModelNet Shape Classification
##### Training/Evaluation
1. Open terminal. Go to classification folder.
2. run:
```
bash training_cmd.sh
```
Alternatively, modify the parameters sent to train.py script to customize your training process.
3. The training script should save trained models and log files into folder "log/your_experiment_name/". In the meantime, evaluation on test set is also carried on during the training, with eval results saved as well.
4. Refer to log_max_record.txt for the model's performance on the eval set. This text file records the best performance (recognition accuracy) of this model during the training and also saves the network parameters into file model_max_record.ckpt. Alternatively you can use tensorboard on the directory "log/your_experiment_name" to monitor the training process. 

##### Example Model: 
1. "pointnet2_cls_ssg.py" : original pointnet++ model, with 4 layer structure as described in our paper.
2. "pointnet2_cls_ssg_spec_cp.py" : our spectral graph convolution model with cluster pooling, using the same 4 layer specification.

You are welcome to try out and test these models. You should be able to demonstrate a noticeable advantage by using our "spec+cp" model when compared with the original pointnet++ model. 

#### - ShapeNet Part Segmentation
Follow ModelNet instructions, and you should be able to run the experiments smoothly. 


