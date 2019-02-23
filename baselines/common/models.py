import numpy as np
import tensorflow as tf
from baselines.a2c import utils
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch
from baselines.common.mpi_running_mean_std import RunningMeanStd
import tensorflow.contrib.layers as layers


def nature_cnn(unscaled_images, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))


def mlp(num_layers=2, num_hidden=64, activation=tf.tanh, **mlp_kwargs):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)
    
    num_hidden: int                 size of fully-connected layers (default: 64)
    
    activation:                     activation function (default: tf.tanh)
        
    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """        
    def network_fn(X):
        h = tf.layers.flatten(X)
        for i in range(num_layers):
            h = activation(fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2)))

        return h, None

    return network_fn

def cnn_mlp(num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False, resolution=24, feat_size = 256, proprio_dim=13, **mlp_kwargs):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    resolution:                     resolution of the image (assumed squared)

    feat_size:                      size of the intermediate feature representation for both the image and the proprioception

    proprio_dim:                    dimensions of the vector that contain proprioceptive information (13 is 3 for ee position, 4 for ee ori quat, and 6 for ee vel)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """


    def network_fn(X): 
            
        rgb = tf.reshape(X[:, :(resolution*resolution*3)], [-1, resolution, resolution, 3])
        proprio = X[:, -proprio_dim:]
        #h_vis = cnn_small()(rgb)

        h_vis = tf.cast(rgb, tf.float32) / 255.
        
        activ = tf.nn.relu
        h_vis = activ(conv(h_vis, 'c1', nf=8, rf=8, stride=4, init_scale=np.sqrt(2)))
        h_vis = activ(conv(h_vis, 'c2', nf=16, rf=4, stride=2, init_scale=np.sqrt(2)))
        h_vis = conv_to_fc(h_vis)
        h_vis = activ(fc(h_vis, 'fc1', nh=feat_size, init_scale=np.sqrt(2)))

        h_prop = tf.layers.flatten(proprio)
        h_prop = activation(fc(h_prop, 'mlpprop_fc1', nh=feat_size, init_scale=np.sqrt(2)))

        h = tf.concat([h_vis, h_prop], 1)
        for i in range(num_layers):
            h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            h = activation(h)

        return h, None #tf.concat([h, h_vis], 1)

    return network_fn

def double_cnn_mlp(num_layers=2, num_hidden=64, activation=tf.tanh, layer_norm=False, resolution=24, feat_size = 256, proprio_dim=13, **mlp_kwargs):
    """
    Stack of fully-connected layers to be used in a policy / q-function approximator

    Parameters:
    ----------

    num_layers: int                 number of fully-connected layers (default: 2)

    num_hidden: int                 size of fully-connected layers (default: 64)

    activation:                     activation function (default: tf.tanh)

    resolution:                     resolution of the image (assumed squared)

    feat_size:                      size of the intermediate feature representation for both the image and the proprioception

    proprio_dim:                    dimensions of the vector that contain proprioceptive information (13 is 3 for ee position, 4 for ee ori quat, and 6 for ee vel)

    Returns:
    -------

    function that builds fully connected network with a given input tensor / placeholder
    """
    def network_fn(X): 
            
        rgb_size = (resolution*resolution*3)
        rgb_one = tf.reshape(X[:, :rgb_size], [-1, resolution, resolution, 3])
        proprio_one = X[:, rgb_size:rgb_size + proprio_dim]

        rgb_two = tf.reshape(X[:, rgb_size+proprio_dim:rgb_size+proprio_dim+rgb_size], [-1, resolution, resolution, 3])
        proprio_two = X[:, rgb_size+proprio_dim+rgb_size:rgb_size+proprio_dim+rgb_size + proprio_dim]
        #h_vis = cnn_small()(rgb)

        h_vis_one = tf.cast(rgb_one, tf.float32) / 255.        
        activ_one = tf.nn.relu
        h_vis_one = activ_one(conv(h_vis_one, 'c11', nf=8, rf=8, stride=4, init_scale=np.sqrt(2)))
        h_vis_one = activ_one(conv(h_vis_one, 'c12', nf=16, rf=4, stride=2, init_scale=np.sqrt(2)))
        h_vis_one = conv_to_fc(h_vis_one)
        h_vis_one = activ_one(fc(h_vis_one, 'fc11', nh=feat_size, init_scale=np.sqrt(2)))

        h_prop_one = tf.layers.flatten(proprio_one)
        h_prop_one = activation(fc(h_prop_one, 'mlpprop_fc11', nh=feat_size, init_scale=np.sqrt(2)))

        h_vis_two = tf.cast(rgb_two, tf.float32) / 255.        
        activ_two = tf.nn.relu
        h_vis_two = activ_two(conv(h_vis_two, 'c21', nf=8, rf=8, stride=4, init_scale=np.sqrt(2)))
        h_vis_two = activ_two(conv(h_vis_two, 'c22', nf=16, rf=4, stride=2, init_scale=np.sqrt(2)))
        h_vis_two = conv_to_fc(h_vis_two)
        h_vis_two = activ_two(fc(h_vis_two, 'fc21', nh=feat_size, init_scale=np.sqrt(2)))

        h_prop_two = tf.layers.flatten(proprio_two)
        h_prop_two = activation(fc(h_prop_two, 'mlpprop_fc21', nh=feat_size, init_scale=np.sqrt(2)))

        h = tf.concat([h_vis_one, h_prop_one,h_vis_two, h_prop_two], 1)
        for i in range(2*num_layers):
            h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                h = tf.contrib.layers.layer_norm(h, center=True, scale=True)
            h = activation(h)

        return h, None #tf.concat([h, h_vis], 1)

    return network_fn
  

def cnn(**conv_kwargs):
    def network_fn(X):
        return nature_cnn(X, **conv_kwargs), None
    return network_fn

def cnn_small(**conv_kwargs):
    def network_fn(X):
        h = tf.cast(X, tf.float32) / 255.
        
        activ = tf.nn.relu
        h = activ(conv(h, 'c1', nf=8, rf=8, stride=4, init_scale=np.sqrt(2), **conv_kwargs))
        h = activ(conv(h, 'c2', nf=16, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
        h = conv_to_fc(h)
        h = activ(fc(h, 'fc1', nh=128, init_scale=np.sqrt(2)))
        return h, None
    return network_fn



def lstm(nlstm=128, layer_norm=False, **mlp_kwargs):
    """
    Builds LSTM (Long-Short Term Memory) network to be used in a policy.
    Note that the resulting function returns not only the output of the LSTM 
    (i.e. hidden state of lstm for each step in the sequence), but also a dictionary
    with auxiliary tensors to be set as policy attributes. 

    Specifically, 
        S is a placeholder to feed current state (LSTM state has to be managed outside policy)
        M is a placeholder for the mask (used to mask out observations after the end of the episode, but can be used for other purposes too)
        initial_state is a numpy array containing initial lstm state (usually zeros)
        state is the output LSTM state (to be fed into S at the next call)


    An example of usage of lstm-based policy can be found here: common/tests/test_doc_examples.py/test_lstm_example
            
    Parameters:
    ----------

    nlstm: int          LSTM hidden state size

    layer_norm: bool    if True, layer-normalized version of LSTM is used

    Returns:
    -------

    function that builds LSTM with a given input tensor / placeholder
    """
        
    def network_fn(X, nenv=1):
        nbatch = X.shape[0] 
        nsteps = nbatch // nenv
         
        h = tf.layers.flatten(X)

        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, 2*nlstm]) #states

        xs = batch_to_seq(h, nenv, nsteps)
        ms = batch_to_seq(M, nenv, nsteps)

        if layer_norm:
            h5, snew = utils.lnlstm(xs, ms, S, scope='lnlstm', nh=nlstm)
        else:
            h5, snew = utils.lstm(xs, ms, S, scope='lstm', nh=nlstm)
            
        h = seq_to_batch(h5)
        initial_state = np.zeros(S.shape.as_list(), dtype=float)

        return h, {'S':S, 'M':M, 'state':snew, 'initial_state':initial_state}

    return network_fn


def cnn_lstm(nlstm=128, layer_norm=False, **conv_kwargs):
    def network_fn(X, nenv=1):
        nbatch = X.shape[0] 
        nsteps = nbatch // nenv
         
        h = nature_cnn(X, **conv_kwargs)
       
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, 2*nlstm]) #states

        xs = batch_to_seq(h, nenv, nsteps)
        ms = batch_to_seq(M, nenv, nsteps)

        if layer_norm:
            h5, snew = utils.lnlstm(xs, ms, S, scope='lnlstm', nh=nlstm)
        else:
            h5, snew = utils.lstm(xs, ms, S, scope='lstm', nh=nlstm)
            
        h = seq_to_batch(h5)
        initial_state = np.zeros(S.shape.as_list(), dtype=float)

        return h, {'S':S, 'M':M, 'state':snew, 'initial_state':initial_state}

    return network_fn

def cnn_lnlstm(nlstm=128, **conv_kwargs):
    return cnn_lstm(nlstm, layer_norm=True, **conv_kwargs)


def conv_only(convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], **conv_kwargs):
    ''' 
    convolutions-only net

    Parameters:
    ----------

    conv:       list of triples (filter_number, filter_size, stride) specifying parameters for each layer. 

    Returns:

    function that takes tensorflow tensor as input and returns the output of the last convolutional layer
    
    '''

    def network_fn(X):
        out = tf.cast(X, tf.float32) / 255.
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu,
                                           **conv_kwargs)

        return out, None
    return network_fn

def _normalize_clip_observation(x, clip_range=[-5.0, 5.0]):
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std, min(clip_range), max(clip_range))
    return norm_x, rms
    

def get_network_builder(name):
    # TODO: replace with reflection? 
    if name == 'cnn':
        return cnn
    elif name == 'cnn_small':
        return cnn_small
    elif name == 'conv_only':
        return conv_only
    elif name == 'mlp':
        return mlp
    elif name == 'lstm':
        return lstm
    elif name == 'cnn_lstm':
        return cnn_lstm
    elif name == 'cnn_lnlstm':
        return cnn_lnlstm
    elif name=='cnn_mlp':
        return cnn_mlp
    elif name=='double_cnn_mlp':
        return double_cnn_mlp
    else:
        raise ValueError('Unknown network type: {}'.format(name))
