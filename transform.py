import cntk as C

from cntk.initializer import he_normal
from cntk.layers import AveragePooling, BatchNormalization, Convolution, Dense
from cntk.ops import element_times, relu, tanh

def _conv_layer(input, num_filters, filter_size, stride, relu=True):
    c = Convolution((filter_size,filter_size), num_filters, activation=None, pad=True, strides=(stride,stride))(input)    
    r = BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=False)(c)
    if relu==True:
        return relu(r)
    else:
        return r

def _conv_tranpose_layer(input, num_filters, filter_size, strides):
    c = ConvolutionTranspose((filter_size,filter_size), num_filters, activation=None, pad=True, strides=(stride,stride))(input)    
    r = BatchNormalization(map_rank=1, normalization_time_constant=4096, use_cntk_engine=False)(c)
    return relu(r)

def _residual_block(input, filter_size=3):
    tmp = _conv_layer(input, 128, filter_size, 1)
    return input + _conv_layer(tmp, 128, filter_size, 1, relu=False)

#   
# Defines the residual network model for image transformation
#
def create_image_transformation_model(input, num_stack_layers, num_classes):
    conv1 = _conv_layer(input, 32, 9, 1)
    conv2 = _conv_layer(conv1, 64, 3, 2)
    conv3 = _conv_layer(conv2, 128, 3, 2)
    resid1 = _residual_block(conv3, 3)
    resid2 = _residual_block(resid1, 3)
    resid3 = _residual_block(resid2, 3)
    resid4 = _residual_block(resid3, 3)
    resid5 = _residual_block(resid4, 3)
    conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2)
    conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2)
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False)
    preds = tanh(conv_t3) * 150 + 255./2
    return preds