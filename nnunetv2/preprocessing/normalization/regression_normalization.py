from nnunetv2.preprocessing.normalization.default_normalization_schemes import ImageNormalization
import numpy as np


class GlobalNormalization(ImageNormalization):
    """
    Global dataset normalization without clipping.
    Uses global mean and std from dataset fingerprint but doesn't clip values.
    """
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, "GlobalNormalization requires intensity properties"
        mean_intensity = self.intensityproperties['mean']
        std_intensity = self.intensityproperties['std']

        image = image.astype(self.target_dtype, copy=False)
        # No clipping - just subtract mean and divide by std
        image -= mean_intensity
        image /= max(std_intensity, 1e-8)
        return image