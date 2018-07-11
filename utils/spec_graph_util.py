import numpy as np
import tensorflow as tf
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling_nd'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point,select_top_k
from tf_interpolate import three_nn, three_interpolate



from tf_util import conv2d, batch_norm_for_conv2d, _variable_with_weight_decay

def corv_mat_setdiag_zero(adj_mat):
    # set diagnal entries of adj mat to 0
    in_shape = adj_mat.get_shape().as_list()

    adj_mat_set_diag = tf.zeros([in_shape[0], in_shape[1] , in_shape[2] ])
    adj_mat = tf.matrix_set_diag(adj_mat, adj_mat_set_diag)

    return adj_mat


def corv_mat_laplacian(adj_mat , flag_normalized = False):
    if not flag_normalized:
        D = tf.reduce_sum(adj_mat , axis = 3)
        D = tf.matrix_diag( D )
        #laplacian matrix
        L = D - adj_mat
    else:
        D = tf.reduce_sum(adj_mat , axis = 3)
        D_sqrt = tf.divide(1.0 , tf.sqrt(D) + 1e-8)
        D = tf.matrix_diag( D )
        D_sqrt = tf.matrix_diag( D_sqrt )

        L = tf.matmul(D_sqrt,
                      tf.matmul( D - adj_mat,
                               D_sqrt
                            )
                    )

    return L

# should be the same as corv_mat_laplacian
def corv_mat_laplacian0(adj_mat , flag_normalized = False):
    if not flag_normalized:
        D = tf.reduce_sum(adj_mat , axis = 3)
        D = tf.matrix_diag( D )
        #laplacian matrix
        L = D - adj_mat
    else:
        D = tf.reduce_sum(adj_mat , axis = 3)
        D_sqrt = tf.divide(1.0 , tf.sqrt(D))
        D_sqrt = tf.matrix_diag( D_sqrt )

        I = tf.ones_like(D , dtype = tf.float32)
        I = tf.matrix_diag( I )
        L = I - tf.matmul(D_sqrt , tf.matmul( adj_mat , D_sqrt))

    return L

def corv_mat_diffusion(adj_mat):

    D = tf.reduce_sum(adj_mat , axis = 3)
    D = 1/D
    D = tf.matrix_diag( D )

    L = tf.matmul(D , adj_mat)

    return L

# fuction for construction knn graph
# input: complete graph's adjacency matrix
def cov_mat_k_nn_graph(adj_mat, k = 3):
    # adj_mat : B N K K
    adj_sorted , adj_sort_ind = tf.nn.top_k(input = adj_mat , k = k, sorted=True)
    adj_thresh = adj_sorted[:,:,:, k - 1] # k-th largest ele

    k_nn_adj_mat = tf.where( tf.less(adj_mat , tf.expand_dims( adj_thresh , axis = -1 ) ),
                        x = tf.zeros_like(adj_mat),
                        y = adj_mat,
                        name = 'k_nn_adj_mat'
                        )

    return k_nn_adj_mat


# euclidean distance as adjacency weights
def get_adj_mat_dist_euclidean(local_cord , flag_normalized = False):
    #print('computing dist based adj_mat')
    # B N K m
    in_shape = local_cord.get_shape().as_list()
    #https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
    # dist between points

    loc_matmul = tf.matmul(local_cord , local_cord, transpose_b=True)
    loc_norm = local_cord * local_cord # B N K m

    r = tf.reduce_sum(loc_norm , -1, keep_dims = True) # B N K 1
    r_t = tf.transpose(r, [0,1,3,2]) # B N 1 K
    D = r - 2*loc_matmul + r_t

    D = tf.identity(D, name='adj_D')

    if flag_normalized:
        D_max = tf.reduce_max( tf.reshape(D , [in_shape[0] , in_shape[1] , in_shape[2] * in_shape[2]]) , axis = -1 )
        D_max = tf.expand_dims(D_max , -1)
        D_max = tf.expand_dims(D_max , -1)
        D_max = tf.tile(D_max , [1,1,in_shape[2],in_shape[2]])
        D = tf.divide(D , D_max + 1e-8)

    # exponential encoding
    # avoid extreme values
    adj_mat = tf.exp(-D, name = 'adj_mat')

    return adj_mat


# cosine distance as adj weights
def get_adj_mat_cos( local_cord , flag_normalized = False, order = 1):
    #print('computing cosine based adj_mat')
    # B n K m
    in_shape = local_cord.get_shape().as_list()
    #print(in_shape)
    #https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
    # dist between points

    loc_matmul = tf.matmul(local_cord , local_cord, transpose_b=True, name = 'loc_matmul') # B n K K

    loc_norm = tf.norm(local_cord , axis = -1, keep_dims=True) # B,n , K, 1
    loc_norm_matmul = tf.matmul(loc_norm , loc_norm, transpose_b=True, name = 'loc_norm_matmul') # B n K K
    D = tf.divide(loc_matmul , loc_norm_matmul + 1e-8 , name = 'cos_D')

    # cos : 1 -> very similar
    # cos : 0 -> distinct
    #D = 1 - D # close: 0 ; far: 1
    # D : B N K K
    # + sign because of cos curve has close as 1 far as -1
    D = tf.exp(D*order)

    if flag_normalized:
        D_max = tf.reduce_max( tf.reshape(D , [in_shape[0] , in_shape[1] , in_shape[2] * in_shape[2]]) , axis = -1 )
        D_max = tf.expand_dims(D_max , -1)
        D_max = tf.expand_dims(D_max , -1)
        D_max = tf.tile(D_max , [1,1,in_shape[2],in_shape[2]])
        D = tf.divide(D , D_max + 1e-8)


    adj_mat = D

    return adj_mat



# outdated
def get_adj_mat(local_cord, type = '3_exp' , flag_use_laplacian = False):
    # input is: Batch , N center points, K neighbor , channel
    in_shape = local_cord.get_shape().as_list()
    ### 1st step: calculate adj matrices
    # dim: B, N, K, K place holder
    adj_mat = tf.ones(
                      shape= (in_shape[0], in_shape[1], in_shape[2],in_shape[2]),
                      dtype=tf.float32
                      )

    if type == '3_exp':
        print(local_cord.get_shape().as_list())
        loc = local_cord
        loc_coef = tf.constant(np.array([1]) , dtype=tf.float32)
        loc_norm = tf.norm(loc, axis = -1 , keep_dims = True)
        loc_norm_mat = tf.matmul( loc_norm , tf.transpose(loc_norm , perm = [0,1,3,2] ) ) + 1e-8
        tmp = tf.matmul( loc , tf.transpose(loc , perm = [0,1,3,2] ) )
        tmp = tf.divide(tmp , loc_norm_mat)
        tmp = tf.multiply(loc_coef , tmp)
        tmp = tf.exp(tmp)
        adj_mat = tf.multiply(tmp , adj_mat)

    if type == '3_regular':
        loc = local_cord
        loc_norm = tf.norm(loc, axis = -1 , keep_dims = True)
        loc_norm_mat = tf.matmul( loc_norm , tf.transpose(loc_norm , perm = [0,1,3,2] ) ) + 1e-8
        tmp = tf.matmul( loc , tf.transpose(loc , perm = [0,1,3,2] ) )
        tmp = tf.divide(tmp , loc_norm_mat)
        adj_mat = tf.multiply(tmp , adj_mat)

    if type == '1_exp':
        print(local_cord.get_shape().as_list())
        val = local_cord
        tmp = val - tf.transpose(val , perm = [0,1,3,2] )
        tmp = tf.exp(tf.abs(tmp));
        adj_mat = tf.multiply(tmp , adj_mat)

    if type == '1_regular':
        print(local_cord.get_shape().as_list())
        val = local_cord
        tmp = val - tf.transpose(val , perm = [0,1,3,2] )
        tmp = tf.abs(tmp);
        adj_mat = tf.multiply(tmp , adj_mat)

    flag_use_laplacian = False

    if flag_use_laplacian:
        L = corv_mat_laplacian(adj_mat)
    else:
        L = adj_mat

    return L



# this cluster pooling method is designed to work for post mlp feats
# therefore no intrinsic feat should be send in
def spec_hier_cluster_pool(inputs , pool_method = 'max' , csize = 4, use_dist_adj = False, fast_approx = False, include_eig = False ):
    in_shape = inputs.get_shape().as_list()
    inputs_ = inputs
    
    K = in_shape[2]
    eig_2nd_saved = list()

    while(K > csize and K % csize == 0):
        # in fast version, no eigen reordering is applied.
        # directly pool over near by
        # do eigen only when 1st roud or fast=false
        if (not fast_approx) or (K == in_shape[2]):
            # compute laplacian
            #adj_mat = get_adj_mat(inputs, type = '3_exp' , flag_use_laplacian = False)
            adj_mat = get_adj_mat_cos(inputs)

            L = corv_mat_laplacian(adj_mat , flag_normalized = True)
            egval , egvect = tf.self_adjoint_eig(L)
            # second smallest eigen value's egvect
            # ind = 1
            ind = tf.constant( np.array([1]) ,dtype=tf.int32)
            partition_vect = tf.squeeze( tf.gather(egvect, ind , axis=-1) ) # B N K
            eig_2nd_saved.append(partition_vect)

            # this part of sorting could be also applied to bipartite spectrual clustering
            # i.e. using median separate the 2nd eig vect, result in precise half half clustering.
            eig_2nd_sorted , sort_ind = tf.nn.top_k(input = partition_vect , k=K, sorted=True)
            # B*N , K
            sort_ind = tf.reshape(sort_ind , [in_shape[0] * in_shape[1] , K ])
            # inputs: B N K m -> BN , K,m
            inputs = tf.reshape(inputs , [in_shape[0] * in_shape[1] , K , in_shape[3]])
            # sorted K points according to 2nd eig vect; 1st half , 2nd hal forms 2 clusters
            inputs = gather_point(inputs, sort_ind) # BN , K,m
        else:
            inputs = tf.reshape(inputs , [in_shape[0] * in_shape[1] , K , in_shape[3]])
        # -> BN, m, k/c , c; last dimension to reduce
        inputs = tf.transpose(inputs , perm = [0,2,1])
        inputs = tf.reshape(inputs , [in_shape[0] * in_shape[1] , in_shape[3] , K/csize , csize ])
        # pooling in-cluster; alternate pool method
        if pool_method == 'max':
            inputs = tf.reduce_max(inputs , axis = -1)
            pool_method = 'avg'
        elif pool_method == 'avg':
            inputs = tf.reduce_mean(inputs , axis = -1)
            pool_method = 'max'
        # BN , m, K/c
        K = K/csize
        inputs = tf.reshape(tf.transpose(inputs, perm = [0,2,1]) ,
                            [in_shape[0] , in_shape[1] , K , in_shape[3]] )


    # inputs now have B N K m where K <= csize, reduce on -2 dim
    if pool_method == 'max':
        inputs = tf.reduce_max(inputs , axis = -2, keep_dims = True)
        pool_method = 'avg'
    elif pool_method == 'avg':
        inputs = tf.reduce_mean(inputs , axis = -2, keep_dims = True)
        pool_method = 'max'

    outputs = inputs


    return outputs





def weight_variable(shape, name=None , mean = 0, var = 0.1):
    initial = tf.truncated_normal_initializer(mean, var)
    var = tf.get_variable(name, shape, tf.float32, initializer=initial)

    #tf.summary.histogram(var.op.name, var)
    return var



# spec conv core layer function
def spec_conv2d(inputs,
                num_output_channels,
                scope,
                nn_k = None,
                local_cord = None,
                use_xavier=True,
                stddev=1e-3,
                weight_decay=0.0,
                activation_fn=None,
                bn=False,
                bn_decay=None,
                is_training=None):


    in_shape = inputs.get_shape().as_list()
    
    # get graph adj matrix
    W = get_adj_mat_dist_euclidean(local_cord[:,:,:,0:3] , flag_normalized = True)
    W = tf.identity(W, name='adjmat')

    # construct k nearest neighbor graph if desired
    if nn_k is not None:
        num_neigh = nn_k
        W_knn = cov_mat_k_nn_graph(W, k = num_neigh )
    else:
        W_knn = W

    # set diag to 0
    W_knn = corv_mat_setdiag_zero(W_knn)
    W_knn = tf.identity(W_knn, name='adjmat_knn')

    L = corv_mat_laplacian0(W_knn , flag_normalized = True)
    L = tf.identity(L, name='laplacian')


    ### eigen decomp
    # egvect: | | | | each vertical line is one eigen vect
    # for PSD mat, SVD <-> eigen

    egval , egvect = tf.self_adjoint_eig(L)
    U = egvect
    UT = tf.transpose(U , perm = [0,1,3,2])


    # transform input to fourier domain
    inputs_fourier = tf.matmul( UT , inputs)

    filtered = inputs_fourier

    # feat expansion
    for i, num_out_channel in enumerate(num_output_channels):
        filtered = conv2d(filtered, num_out_channel, [1,1],
                          padding='VALID', stride=[1,1],
                          bn=False, is_training=is_training,
                          scope= scope + 'conv2d_%d'%(i),
                          bn_decay=bn_decay,
                          activation_fn = None)


    outputs = tf.matmul( U , filtered )

    #BN and relu
    outputs = batch_norm_for_conv2d(outputs, is_training, bn_decay=bn_decay, scope='bn_post_spec')
    outputs = tf.nn.relu(outputs)
    
    return outputs, UT




# clean version with spec modulation
def spec_conv2d_modul(inputs,
                num_output_channels,
                scope,
                nn_k = None,
                local_cord = None,
                use_xavier=True,
                stddev=1e-3,
                weight_decay=0.0,
                activation_fn=None,
                bn=False,
                bn_decay=None,
                is_training=None):


    in_shape = inputs.get_shape().as_list()
    
    # get graph adj matrix
    W = get_adj_mat_dist_euclidean(local_cord[:,:,:,0:3] , flag_normalized = True)
    W = tf.identity(W, name='adjmat')

    # construct k nearest neighbor graph if desired
    if nn_k is not None:
        num_neigh = nn_k
        W_knn = cov_mat_k_nn_graph(W, k = num_neigh )
    else:
        W_knn = W

    # set diag to 0
    W_knn = corv_mat_setdiag_zero(W_knn)
    W_knn = tf.identity(W_knn, name='adjmat_knn')

    L = corv_mat_laplacian0(W_knn , flag_normalized = True)
    L = tf.identity(L, name='laplacian')

    ### eigen decomp
    # egvect: | | | | each vertical line is one eigen vect
    # for PSD mat, SVD <-> eigen
    flag_use_svd = False
    if flag_use_svd:
        s, u, v  = tf.svd(
                          L,
                          compute_uv=True
                          )

        U = u
        UT = tf.transpose(v , perm = [0,1,3,2])
        egval = s

    else:
        egval , egvect = tf.self_adjoint_eig(L)
        U = egvect
        UT = tf.transpose(U , perm = [0,1,3,2])


    # transform input to fourier domain
    inputs_fourier = tf.matmul( UT , inputs)

    # spec modulation
    W_modulation = weight_variable(shape = (1,1, in_shape[-2]), name='spec_modulation' + scope , mean = 1.0, var = stddev)
    W_modulation = tf.matrix_diag( W_modulation )
    W_modulation = tf.tile(W_modulation , [ in_shape[0] , in_shape[1] , 1 , 1 ])
    filtered = tf.matmul( W_modulation , inputs_fourier)

    # feat expansion
    for i, num_out_channel in enumerate(num_output_channels):
        filtered = conv2d(filtered, num_out_channel, [1,1],
                          padding='VALID', stride=[1,1],
                          bn=False, is_training=is_training,
                          scope= scope + 'conv2d_%d'%(i),
                          bn_decay=bn_decay,
                          activation_fn = None)


    outputs = tf.matmul( U , filtered )

    #BN and relu
    outputs = batch_norm_for_conv2d(outputs, is_training, bn_decay=bn_decay, scope='bn_post_spec')
    outputs = tf.nn.relu(outputs)
    
    return outputs, UT
