import matplotlib
from packaging.version import Version


if not Version(matplotlib.__version__) < Version('3.11.0'):
    raise ImportError("Workflow currently only supports `matplotlib<3.11.0`")
