# Copyright (c) Wenhui Lu. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from easydict import EasyDict as edict

__C = edict()
cfg = __C
#
# Fast neural style
#

__C.VGG16_MODEL_FILENAME = 'vgg16_weights.bin'
__C.MODEL_FILE_PATH = 'models'
__C.CONTENT_FILE_PATH = 'images/content'
__C.CONTENT_DEFAULT_FILENAME = 'content.jpg'
__C.STYLE_FILE_PATH = 'images/style'
__C.STYLE_DEFAULT_FILENAME = 'style.jpg'
__C.OUTPUT_FILE_PATH = 'images/output'
__C.OUTPUT_DEFAULT_FILENAME = 'output%d.jpg'