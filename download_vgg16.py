# Copyright (c) Wenhui Lu. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
from config import cfg as cfg
from util import download


path = os.path.join(cfg.MODEL_FILE_PATH,cfg.VGG16_MODEL_FILENAME)
url = 'https://cntk.ai/jup/models/vgg16_weights.bin'

# We check for the model locally
if not os.path.exists(path):
    # download the file from the web
    print('downloading VGG model (~500MB)')
    download(url, path)
else:
    print('VGG model already downloaded')