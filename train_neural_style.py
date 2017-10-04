# Copyright (c) Wenhui Lu. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import h5py
import os
import sys
import numpy as np
import cntk as C
import matplotlib.pyplot as plt
from scipy import optimize as opt
from config import cfg as cfg
from util import download
from PIL import Image
C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components

def load_vgg(path):
    f = h5py.File(path)
    layers = []
    for k in range(f.attrs['nb_layers']):
        g = f['layer_{}'.format(k)]
        n = g.attrs['nb_params']
        layers.append([g['param_{}'.format(p)][:] for p in range(n)])
    f.close()
    return layers

# We check for the model locally

path = os.path.join(cfg.MODEL_FILE_PATH,cfg.VGG16_MODEL_FILENAME)
if not os.path.exists(path):
    print(r"Please download VGG-16 model first by running 'download_vgg16.py' first.")
    sys.exit()

layers = load_vgg(path)
print('loaded VGG model')


# A convolutional layer in the VGG network
def vggblock(x, arrays, layer_map, name):
    f = arrays[0]
    b = arrays[1]
    k = C.constant(value=f)
    t = C.constant(value=np.reshape(b, (-1, 1, 1)))
    y = C.relu(C.convolution(k, x, auto_padding=[False, True, True]) + t)
    layer_map[name] = y
    return y

# A pooling layer in the VGG network
def vggpool(x):
    return C.pooling(x, C.AVG_POOLING, (2, 2), (2, 2))


# Build the graph for the VGG network (excluding fully connected layers)
def model(x, layers):
    model_layers = {}
    def convolutional(z): return len(z) == 2 and len(z[0].shape) == 4
    conv = [layer for layer in layers if convolutional(layer)]
    cnt = 0
    num_convs = {1: 2, 2: 2, 3: 3, 4: 3, 5: 3}
    for outer in range(1,6):
        for inner in range(num_convs[outer]):
            x = vggblock(x, conv[cnt], model_layers, 'conv%d_%d' % (outer, 1+inner))
            cnt += 1
        x = vggpool(x)
    return x, C.combine([model_layers[k] for k in sorted(model_layers.keys())])

def flatten(x):
    assert len(x.shape) >= 3
    return C.reshape(x, (x.shape[-3], x.shape[-2] * x.shape[-1]))


def gram(x):
    features = C.minus(flatten(x), C.reduce_mean(x))
    return C.times_transpose(features, features)


def npgram(x):
    features = np.reshape(x, (-1, x.shape[-2]*x.shape[-1])) - np.mean(x)
    return features.dot(features.T)


def style_loss(a, b):
    channels, x, y = a.shape
    assert x == y
    A = gram(a)
    B = npgram(b)
    return C.squared_error(A, B)/(channels**2 * x**4)


def content_loss(a,b):
    channels, x, y = a.shape
    return C.squared_error(a, b)/(channels*x*y)


def total_variation_loss(x):
    xx = C.reshape(x, (1,)+x.shape)
    delta = np.array([-1, 1], dtype=np.float32)
    kh = C.constant(value=delta.reshape(1, 1, 1, 1, 2))
    kv = C.constant(value=delta.reshape(1, 1, 1, 2, 1))
    dh = C.convolution(kh, xx, auto_padding=[False])
    dv = C.convolution(kv, xx, auto_padding=[False])
    avg = 0.5 * (C.reduce_mean(C.square(dv)) + C.reduce_mean(C.square(dh)))
    return avg


style_path = os.path.join(cfg.STYLE_FILE_PATH,cfg.STYLE_DEFAULT_FILENAME)
content_path = os.path.join(cfg.CONTENT_FILE_PATH,cfg.CONTENT_DEFAULT_FILENAME)

start_from_random = False
content_weight = 5.0
style_weight = 1.0
decay = 0.5

outer = 10
inner = 20
SIZE = 300

SHIFT = np.reshape([103.939, 116.779, 123.68], (3, 1, 1)).astype('f')

def load_image(path):
    with Image.open(path) as pic:
        hw = pic.size[0] / 2
        hh = pic.size[1] / 2
        mh = min(hw,hh)
        cropped = pic.crop((hw - mh, hh - mh, hw + mh, hh + mh))
        array = np.array(cropped.resize((SIZE,SIZE), Image.BICUBIC), dtype=np.float32)
        return np.ascontiguousarray(np.transpose(array, (2,0,1)))-SHIFT

def save_image(img, path):
    sanitized_img = np.maximum(0, np.minimum(255, img+SHIFT))
    pic = Image.fromarray(np.uint8(np.transpose(sanitized_img, (1, 2, 0))))
    pic.save(path)

def ordered_outputs(f, binding):
    _, output_dict = f.forward(binding, f.outputs)
    return [np.squeeze(output_dict[out]) for out in f.outputs]

# download the images if they are not available locally
if not os.path.exists(style_path):
    download('https://cntk.ai/jup/%s' % cfg.STYLE_DEFAULT_FILENAME, style_path)
if not os.path.exists(content_path):
    download('https://cntk.ai/jup/%s' % cfg.CONTENT_DEFAULT_FILENAME, content_path)

# Load the images
style   = load_image(style_path)
content = load_image(content_path)

# Display the images
for img in content, style:
    plt.figure()
    plt.imshow(np.asarray(np.transpose(img+SHIFT, (1, 2, 0)), dtype=np.uint8))

# Push the images through the VGG network 
# First define the input and the output
y = C.input_variable((3, SIZE, SIZE), needs_gradient=True)
z, intermediate_layers = model(y, layers)
# Now get the activations for the two images
content_activations = ordered_outputs(intermediate_layers, {y: [[content]]})
style_activations = ordered_outputs(intermediate_layers, {y: [[style]]})
style_output = np.squeeze(z.eval({y: [[style]]}))

# Finally define the loss
n = len(content_activations)
total = (1-decay**(n+1))/(1-decay) # makes sure that changing the decay does not affect the magnitude of content/style
loss = (1.0/total * content_weight * content_loss(y, content) 
         + 1.0/total * style_weight * style_loss(z, style_output) 
         + total_variation_loss(y))

for i in range(n):
    loss = (loss 
        + decay**(i+1)/total * content_weight * content_loss(intermediate_layers.outputs[i], content_activations[i])
        + decay**(n-i)/total * style_weight   *   style_loss(intermediate_layers.outputs[i], style_activations[i]))

# utility to convert a vector to an image
def vec2img(x):
    d = np.round(np.sqrt(x.size / 3)).astype('i')
    return np.reshape(x.astype(np.float32), (3, d, d))

# utility to convert an image to a vector
def img2vec(img):
    return img.flatten().astype(np.float64)

# utility to compute the value and the gradient of f at a particular place defined by binding
def value_and_grads(f, binding):
    if len(f.outputs) != 1:
        raise ValueError('function must return a single tensor')
    df, valdict = f.forward(binding, [f.output], set([f.output]))
    value = list(valdict.values())[0]
    grads = f.backward(df, {f.output: np.ones_like(value)}, set(binding.keys()))
    return value, grads

# an objective function that scipy will be happy with
def objfun(x, loss):
    y = vec2img(x)
    v, g = value_and_grads(loss, {loss.arguments[0]: [[y]]})
    v = np.reshape(v, (1,))
    g = img2vec(list(g.values())[0])
    return v, g

# the actual optimization procedure
def optimize(loss, x0, inner, outer):
    bounds = [(-np.min(SHIFT), 255-np.max(SHIFT))]*x0.size
    for i in range(outer):
        s = opt.minimize(objfun, img2vec(x0), args=(loss,), method='L-BFGS-B', 
                         bounds=bounds, options={'maxiter': inner}, jac=True)
        print('objective : %s' % s.fun[0])
        x0 = vec2img(s.x)
        filename = cfg.OUTPUT_DEFAULT_FILENAME % i
        path = os.path.join(cfg.OUTPUT_FILE_PATH,filename) 
        save_image(x0, path)
    return x0

np.random.seed(98052)
if start_from_random:
    x0 = np.random.randn(3, SIZE, SIZE).astype(np.float32)
else:
    x0 = content
xstar = optimize(loss, x0, inner, outer)
plt.imshow(np.asarray(np.transpose(xstar+SHIFT, (1, 2, 0)), dtype=np.uint8))