from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class SAMModel:
    """
    Wrapper that builds a SAM-2 model and offers convenience helpers.
    """

    def __init__(
        self, 
        model_config: str, 
        checkpoint_path: str, 
        device: str
        ) -> None:
        """
        Description
        -----------
        Construct the wrapper and remember paths / device; 

        Parameters
        ----------
        model_config : str
                YAML cfg describing network architecture.
        checkpoint_path : str
                Path to *.pt checkpoint with learned weights.
        device : str
                Torch device ('cuda' | 'cpu' | 'cuda:0', …).

        Returns
        -------
        None
        """
        self.model_config: str = model_config
        self.checkpoint: str = checkpoint_path
        self.device: str = device

    def setup_cuda(self) -> None:
        """
        Description
        -----------
        Enable bfloat16 and TF32 on Ampere GPUs for speed.

        Parameters;
                none
        Returns
                None
        """
        if torch.cuda.is_available():
            torch.autocast(
                device_type=self.device, 
                dtype=torch.bfloat16
                ).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def _build_model(self) -> "torch.nn.Module":
        """
        Description
        -----------
        Internal helper to build the SAM-2 network.

        Parameters: none
        Returns
        -------
        torch.nn.Module
                SAM-2 network on requested device.
        """
        return build_sam2(
            self.model_config,
            self.checkpoint,
            device=self.device,
            apply_postprocessing=False,
        )

    def generate_masks(
        self,
        output_dir: str | Path,
        image: np.ndarray,
        mask_settings: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Description
        -----------
        Run SAM-2 automatic mask generator on *image*.

        Parameters
        ----------
        output_dir : str | Path
                Folder to dump raw mask PNGs.
        image : np.ndarray
                H x W x 3 RGB image.
        mask_settings : dict | None
                Generator hyper-parameters (*kwargs).

        Returns
        -------
        List[Dict[str, Any]]
                Masks sorted by descending area.
        """
        out_path: Path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        model = self._build_model()
        gen = SAM2AutomaticMaskGenerator(model=model, **(mask_settings or {}))
        masks: List[Dict[str, Any]] = gen.generate(image)

        masks_sorted: List[Dict[str, Any]] = sorted(
            masks, 
            key=lambda m: m["area"], 
            reverse=True
            )
        return masks_sorted

    def mask_summary(
        self, 
        masks: List[Dict[str, Any]]
        ) -> pd.DataFrame:
        """
        Description
        -----------
        Convert list-of-dict masks to a tidy pandas table.

        Parameters
        ----------
        masks : List[Dict[str, Any]]
                Output of generate_masks().

        Returns
        -------
        pd.DataFrame
                One row per mask, all original keys.
        """
        exclude: set[str] = set()
        rows: List[Dict[str, Any]] = [{k: v for k, v in m.items() if k not in exclude} for m in masks]
        return pd.DataFrame(rows)
    