#!/usr/bin/env python3
import numpy as np
from dipy.io.streamline import load_trk
from dipy.segment.bundles import bundle_shape_similarity
import os
import json

if __name__ == '__main__':

    # Create Brainlife's output dirs if don't exist
    if not os.path.exists('output'):
        os.mkdir('output')
    if not os.path.exists('secondary'):
        os.mkdir('secondary')

    # Read Brainlife's config.json
    with open('config.json', encoding='utf-8') as config_json:
        config = json.load(config_json)


    fname1 = config .get('bundle1')
    fname2 = config .get('bundle2')

    bundle1 = load_trk(fname1, reference=fname1,
                       bbox_valid_check=False).streamlines

    bundle2 = load_trk(fname2, reference=fname2,
                       bbox_valid_check=False).streamlines

    threshold = config .get('threshold')

    clust_thr=[0.1]

    rng = np.random.RandomState()

    score = bundle_shape_similarity(bundle1, bundle2, rng, clust_thr,
                                    threshold)


    if os.path.exists('output'):
        np.save("shape_similarity_score.npy", np.array(score))