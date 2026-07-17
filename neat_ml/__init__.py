from packaging.version import Version
from packaging.specifiers import SpecifierSet
from importlib.metadata import version


def _check_dependency_compatibility():
    # initialize dict of supported versions
    supported_versions = {
        "matplotlib": ("<3.11.0", "Workflow currently only supports `matplotlib<3.11.0`"),
        "numpy": (">=1.26.3,<=2.1.3", "Workflow currently only supports `numpy>=1.26.3, <=2.1.3` (see issue #38)"),
        "xgboost": (">=2.1.4", "Workflow currently only supports `xgboost>=2.1.4`"),
        "shap": (">=0.47.0", "Workflow currently only supports `shap>=0.47.0`"),
        "torch": (">=2.5.1", "SAM-2 requires `torch>=2.5.1`"),
        "torchvision": (">=0.21.1", "SAM-2 requires `torchvision>=0.21.1`"),
        "scipy": ("!=1.17.0", "Workflow does not support `scipy==1.17.0`"),
    }
    # iterate through dependency list checking installed
    # versions against supported versions
    for dep, (ver, msg) in supported_versions.items():
        # get installed and supported versions
        installed_version = Version(version(dep))
        supported_version = SpecifierSet(ver)
        # perform version check and raise ImportError with unsupported dependency versions
        if installed_version not in supported_version:
            raise ImportError(f"{msg}. You have version {installed_version} installed.")

_check_dependency_compatibility()
