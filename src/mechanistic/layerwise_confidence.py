import numpy as np


def extract_layerwise_confidence(layer_conf):

    early_conf = np.mean(layer_conf[:5])

    middle_conf = np.mean(
        layer_conf[
            len(layer_conf)//3 :
            2*len(layer_conf)//3
        ]
    )

    late_conf = np.mean(layer_conf[-5:])

    return (
        early_conf,
        middle_conf,
        late_conf
    )
