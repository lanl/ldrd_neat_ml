from typing import Any, Optional

import numpy as np
import torch
import logging

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

logger = logging.getLogger(__name__)


class SAMModel:
    """
    Wrapper that builds a SAM-2 model and offers convenience helpers.
    """

    def __init__(
        self, 
        checkpoint_path: str, 
        device: str,
        mask_settings: Optional[dict[str, Any]] = None,
        ) -> None:
        """
        Description
        -----------
        Construct the wrapper, build the SAM2 model, and remember paths / device; 

        Parameters
        ----------
        checkpoint_path : str
                Path to model checkpoint with learned weights.
        device : str
                Torch device ('cuda' | 'cpu' | 'mps' | 'cuda:0', …).
        mask_settings : dict | None
                Generator hyper-parameters (*kwargs).
        """
        self.mask_settings = mask_settings
        self.checkpoint = checkpoint_path
        self.device = self.setup_cuda(device)
        self.model = self._build_model()

    def setup_cuda(self, device: str) -> str:
        """
        Description
        -----------
        Determine the appropriate device for running SAM-2 detection
        based on user input and available hardware.

        Enable bfloat16 and TF32 on Ampere GPUs for speed or
        log warning when using `mps` backend on macos
        
        Note: this code is derived from the example jupyter notebook
        stored at:

        https://github.com/facebookresearch/sam2/blob/main/
        notebooks/automatic_mask_generator_example.ipynb
        """
        if device == "gpu":
            if torch.cuda.is_available():
                torch.autocast(
                    device_type=self.device, 
                    dtype=torch.bfloat16
                    ).__enter__()
                # turn on tfloat32 for Ampere GPUs 
                # (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                if torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            elif torch.backends.mps.is_available():
                logger.warning(
                    "Support for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                    "give numerically different outputs and sometimes degraded performance on MPS. "
                    "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
                )
        self.device = (
            "cuda" if (torch.cuda.is_available() and device == "gpu")
            else "mps" if (torch.backends.mps.is_available() and device == "gpu")
            else "cpu"
        )
        return self.device

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
        self._model = SAM2AutomaticMaskGenerator.from_pretrained(
            self.checkpoint,
            device=self.device,
            apply_postprocessing=False,
            **(self.mask_settings or {}))
        return self._model

    def generate_masks(
        self,
        image: np.ndarray,
    ) -> list[dict[str, Any]]:
        """
        Description
        -----------
        Run SAM-2 automatic mask generator on *image*.

        Parameters
        ----------
        image : np.ndarray
                H x W x 3 RGB image.

        Returns
        -------
        masks_sorted : list[dict[str, Any]]
                Masks sorted by descending area.
        """

        masks = self._model.generate(image)

        masks_sorted = sorted(
            masks, 
            key=lambda m: m["area"], 
            reverse=True
            )
        return masks_sorted 
