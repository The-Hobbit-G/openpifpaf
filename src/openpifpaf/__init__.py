"""An open implementation of PifPaf."""

# pylint: disable=wrong-import-position

from . import _version
__version__ = _version.get_versions()['version']

# register ops first
from . import cpp_extension
cpp_extension.register_ops()

from .annotation import Annotation, AnnotationDet
from .configurable import Configurable
from .predictor import Predictor
from .signal import Signal
from . import datasets
from . import decoder
from . import encoder
from . import headmeta
from . import logger
from . import metric
from . import network
from . import optimize
from . import plugin
from . import visualizer

from .datasets import DATAMODULES
from .decoder import DECODERS
from .network.factory import (
    BASE_FACTORIES,
    BASE_TYPES,
    CHECKPOINT_URLS,
    HEADS,
    PRETRAINED_UNAVAILABLE,
)
from .network.losses.factory import LOSSES, LOSS_COMPONENTS
from .network.model_migration import MODEL_MIGRATION
from .show.annotation_painter import PAINTERS

# load plugins last
plugin.register()


##Since in the training and evaluation script, we call 'python3 -m openpifpaf.eval ...' 
##When you use the -m option to run a Python module as a script, Python first imports the openpifpaf module as a package, 
##which means it executes this __init__.py file. Then it looks for a module or script named eval inside the package and executes it.
##So all the codes in this __init__.py file will be executed and since we have plugin.register() here, 
##DATAMODULES from .datasets will be inserted the corresponding datamodule and won't be empaty