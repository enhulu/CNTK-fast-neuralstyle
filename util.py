# Copyright (c) Wenhui Lu. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import requests

def download(url, filename):
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as handle:
        for data in response.iter_content(chunk_size=2**20):
            if data: handle.write(data)