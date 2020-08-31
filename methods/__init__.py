from .infonce import InfoNCE
from .w_mse import WMSE
from .byol import BYOL


METHOD_LIST = ["nce", "w_mse", "byol"]


def get_method(name):
    assert name in METHOD_LIST
    if name == "nce":
        return InfoNCE
    elif name == "w_mse":
        return WMSE
    elif name == "byol":
        return BYOL
