from .contrastive import Contrastive
from .w_mse import WMSE
from .byol import BYOL


METHOD_LIST = ["contrastive", "w_mse", "byol"]


def get_method(name):
    assert name in METHOD_LIST
    if name == "contrastive":
        return Contrastive
    elif name == "w_mse":
        return WMSE
    elif name == "byol":
        return BYOL
