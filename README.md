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


