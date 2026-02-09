from typing import Any, Optional

import numpy as np
import torch
import logging
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from importlib.resources.abc import Traversable

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

logger = logging.getLogger(__name__)


class SAMModel:
    """
    Wrapper that builds a SAM-2 model and offers convenience helpers.
    """

    def __init__(
        self, 
        model_config: str, 
        checkpoint_path: Traversable, 
        device: str
        ) -> None:
        """
        Description
        -----------
        Construct the wrapper, build the SAM2 model, and remember paths / device; 

        Parameters
        ----------
        model_config : str
                Path to YAML cfg describing network architecture.
        checkpoint_path : Traversable
                Path to *.pt checkpoint with learned weights.
        device : str
                Torch device ('cuda' | 'cpu' | 'mps' | 'cuda:0', …).
        """
        self.model_config = model_config
        self.checkpoint = checkpoint_path
        self.device = device
        self.model = self._build_model()

    def setup_cuda(self) -> None:
        """
        Description
        -----------
        Enable bfloat16 and TF32 on Ampere GPUs for speed or
        log warning when using `mps` backend on macos
        
        Note: this code is derived from the example jupyter notebook
        stored at:

        https://github.com/facebookresearch/sam2/blob/main/
        notebooks/automatic_mask_generator_example.ipynb
        """
        if torch.cuda.is_available():
            torch.autocast(
                device_type=self.device, 
                dtype=torch.bfloat16
                ).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif torch.backends.mps.is_available():
            logger.warning(
                "Support for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )

    def _build_model(self) -> torch.nn.Module:
        """
        Description
        -----------
        Internal helper to build the SAM-2 network.

        Returns
        -------
        torch.nn.Module
                SAM-2 network on requested device.
        """
        self._model = build_sam2(
            self.model_config,
            self.checkpoint,
            device=self.device,
            apply_postprocessing=False,
        )
        return self._model

    def generate_masks(
        self,
        image: np.ndarray,
        mask_settings: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Description
        -----------
        Run SAM-2 automatic mask generator on *image*.

        Parameters
        ----------
        image : np.ndarray
                H x W x 3 RGB image.
        mask_settings : dict | None
                Generator hyper-parameters (*kwargs).

        Returns
        -------
        masks_sorted : list[dict[str, Any]]
                Masks sorted by descending area.
        """

        gen = SAM2AutomaticMaskGenerator(model=self._model, **(mask_settings or {}))
        masks = gen.generate(image)

        masks_sorted = sorted(
            masks, 
            key=lambda m: m["area"], 
            reverse=True
            )
        return masks_sorted 
