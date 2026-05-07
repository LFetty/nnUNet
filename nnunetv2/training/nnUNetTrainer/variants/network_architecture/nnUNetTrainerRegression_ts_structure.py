"""
Trainer variant that adds a TotalSegmentator-based structure + feature loss.

Workflow at init
----------------
1. Load the TS model (via ``TSStructureLoss``) onto the training device.
2. Preload all cached CT-side TS labels and features into an in-memory dict
   keyed by case id. Sizes are modest for typical thorax/abdomen/head-neck
   datasets (~50–100 cases × int8 labels + fp16 features).

Workflow at each train step
---------------------------
1. Run the usual regression forward + compound loss (MAE/grad/perceptual etc.).
2. Look up cached labels + features per case in the batch, crop to the
   patch bbox, stack into a batch tensor.
3. Forward the frozen TS model on the sCT prediction with hooks, compute
   DC+CE against cached labels + L1 against cached features.
4. Add the weighted TS structure term to the base loss.

Expected inputs
---------------
This trainer relies on a data loader that exposes ``bbox`` (patch start/end
per sample) and ``keys`` (case ids) in every yielded batch. For v1 we keep the
implementation minimal: we override ``train_step`` only and read these from
``batch['bbox']`` / ``batch['keys']``.  See ``nnUNetDataLoader3D_TSStructure``
for a one-line data-loader subclass that adds ``bbox`` to the yielded dict.
"""
from __future__ import annotations

import json as _json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import blosc2
import numpy as np
import torch
from torch import autocast

from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.loss.ts_structure_loss import TSStructureLoss, _resolve_ts_trainer_folder
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerRegression_advanced import (
    nnUNetTrainerRegression_advanced,
)
from nnunetv2.utilities.helpers import dummy_context


# ---------------------------------------------------------------------------
# Segmentation evaluation region configs
# Add new regions here; each entry is the list of TotalSegmentator structure
# names to include in the Dice evaluation for that body region.
# ---------------------------------------------------------------------------
REGION_SEG_CONFIGS: Dict[str, List[str]] = {
    "thorax": [
        "lung_upper_lobe_left",
        "lung_lower_lobe_left",
        "lung_upper_lobe_right",
        "lung_middle_lobe_right",
        "lung_lower_lobe_right",
        "trachea",
        "heart",
        "aorta",
        "esophagus",
        "sternum",
    ],
    # Future regions:
    # "abdomen": [...],
    # "head_neck": [...],
}


def _crop_and_pad_3d(full, bbox, patch_shape, pad_value=0):
    """Crop `full` (numpy or blosc2) with a 3D bbox, padding with `pad_value`."""
    import numpy as _np
    lo = [b[0] for b in bbox]
    hi = [b[0] + patch_shape[d] for d, b in enumerate(bbox)]
    lo_c = [max(0, v) for v in lo]
    hi_c = [min(full.shape[d], v) for d, v in enumerate(hi)]
    crop = _np.asarray(full[lo_c[0]:hi_c[0], lo_c[1]:hi_c[1], lo_c[2]:hi_c[2]])
    pads = [(lo_c[d] - lo[d], hi[d] - hi_c[d]) for d in range(3)]
    if any(p != (0, 0) for p in pads):
        crop = _np.pad(crop, pads, mode="constant", constant_values=pad_value)
    return crop


class _nnUNetDataLoaderWithBbox(nnUNetDataLoader):
    """Data loader that returns bbox + preloaded TS cache crops in every batch.

    TS cache reads happen in the worker process, overlapped with GPU training
    (same as the preprocessed data). This is critical on slow filesystems
    (e.g. /mnt/c in WSL2) where a single blosc2 slice can cost 200+ ms.
    """

    _ts_cache_dir: Optional[Path] = None
    _ts_feature_stages: List[str] = []
    _ts_feat_strides: Dict[str, Tuple[int, int, int]] = {}

    @classmethod
    def set_ts_cache(cls, cache_dir, stages, strides):
        cls._ts_cache_dir = Path(cache_dir)
        cls._ts_feature_stages = list(stages)
        cls._ts_feat_strides = dict(strides)

    def get_bbox(self, *args, **kwargs):
        lbs, ubs = super().get_bbox(*args, **kwargs)
        self._pending_bboxes.append([(int(lbs[d]), int(ubs[d])) for d in range(len(lbs))])
        return lbs, ubs

    def generate_train_batch(self):
        self._pending_bboxes: List[List[Tuple[int, int]]] = []
        out = super().generate_train_batch()
        out["bbox"] = self._pending_bboxes

        if self._ts_cache_dir is None:
            return out

        # Lazy per-worker handle caches (empty after fork → repopulated here).
        if not hasattr(self, "_lbl_handles"):
            self._lbl_handles: Dict[str, object] = {}
            self._feat_handles: Dict[str, Dict[str, object]] = {}

        keys = [str(k) for k in out["keys"]]
        patch_shape = tuple(out["data"].shape[-3:])
        B = len(keys)

        # Labels
        ts_labels = np.zeros((B, 1, *patch_shape), dtype=np.int64)
        for b, (key, bbox) in enumerate(zip(keys, self._pending_bboxes)):
            h = self._lbl_handles.get(key)
            if h is None:
                h = blosc2.open(urlpath=str(self._ts_cache_dir / f"{key}_ts_labels.b2nd"), mode="r")
                self._lbl_handles[key] = h
            ts_labels[b, 0] = _crop_and_pad_3d(h, bbox, patch_shape, pad_value=0)
        out["ts_labels"] = ts_labels

        # Features (per stage)
        ts_feats: Dict[str, np.ndarray] = {}
        for stage in self._ts_feature_stages:
            stride = self._ts_feat_strides[stage]
            feat_patch = tuple(max(1, patch_shape[d] // stride[d]) for d in range(3))
            per = []
            for key, bbox in zip(keys, self._pending_bboxes):
                feat_for_key = self._feat_handles.setdefault(key, {})
                h = feat_for_key.get(stage)
                if h is None:
                    h = blosc2.open(urlpath=str(self._ts_cache_dir / f"{key}_ts_feat_{stage}.b2nd"), mode="r")
                    feat_for_key[stage] = h
                lo = [bbox[d][0] // stride[d] for d in range(3)]
                hi = [lo[d] + feat_patch[d] for d in range(3)]
                lo_c = [max(0, v) for v in lo]
                hi_c = [min(h.shape[d + 1], v) for d, v in enumerate(hi)]
                crop = np.asarray(h[:, lo_c[0]:hi_c[0], lo_c[1]:hi_c[1], lo_c[2]:hi_c[2]])
                pads = [(0, 0)] + [(lo_c[d] - lo[d], hi[d] - hi_c[d]) for d in range(3)]
                if any(p != (0, 0) for p in pads):
                    crop = np.pad(crop, pads, mode="constant", constant_values=0.0)
                per.append(crop)
            ts_feats[stage] = np.stack(per, axis=0).astype(np.float32)
        out["ts_feat"] = ts_feats
        return out


def _open_b2nd(path: Path) -> np.ndarray:
    """Open a blosc2 file and return it as a lazy array (indexable like numpy)."""
    return blosc2.open(urlpath=str(path), mode="r")


class nnUNetTrainerRegression_ts_structure(nnUNetTrainerRegression_advanced):
    """
    Regression trainer with compound loss + TS structure/feature loss.

    Environment variables expected:
        TS_RESULTS_DIR       : path to the TS nnUNet results folder
        TS_CACHE_DIR         : directory written by scripts/cache_ts_targets.py
        TS_CONFIG            : TS nnUNet config name (default: 3d_fullres)
        TS_FOLD              : TS fold (default: 0)

    Class attributes (override for ablations):
        w_ts_seg  : weight on DC+CE(sCT, cached_labels). Default 0.1.
        w_ts_feat : weight on L1(sCT_feat, cached_feat).  Default 0.1.
        ts_feature_stages : encoder stages to hook. Default ["bottleneck", "mid"].
    """

    w_ts_seg: float = 0.1
    w_ts_feat: float = 0.1
    ts_feature_stages: List[str] = ["bottleneck", "mid"]
    ts_eval_region: str = "thorax"

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self._ts_results_dir = os.environ.get("TS_RESULTS_DIR")
        self._ts_cache_dir = os.environ.get("TS_CACHE_DIR")
        self._ts_config = os.environ.get("TS_CONFIG", "3d_fullres")
        self._ts_fold = int(os.environ.get("TS_FOLD", "0"))
        # Optional: pre-cached GT TS segmentations produced by
        #   cache_ts_targets.py --eval_gt_dir ... --eval_cache_dir ...
        # When set, GT segs are loaded from disk instead of re-running TS at eval time.
        self._ts_eval_cache_dir: Optional[str] = os.environ.get("TS_EVAL_CACHE_DIR")

        if not self._ts_results_dir or not self._ts_cache_dir:
            raise ValueError(
                "TS_RESULTS_DIR and TS_CACHE_DIR env vars must be set to use "
                "nnUNetTrainerRegression_ts_structure."
            )

        # Target-channel normalization stats (training-space → HU)
        normalization_schemes = self.configuration_manager.normalization_schemes
        ct_idx = 1 if len(normalization_schemes) > 1 else 0
        tp = self.plans_manager.foreground_intensity_properties_per_channel[str(ct_idx)]
        target_mean = float(tp.get("mean", 0.0))
        target_std = float(tp.get("std", 1.0))

        self.ts_loss = TSStructureLoss(
            ts_results_dir=self._ts_results_dir,
            ts_config=self._ts_config,
            ts_fold=self._ts_fold,
            feature_stages=self.ts_feature_stages,
            target_mean=target_mean,
            target_std=target_std,
            w_seg=1.0,
            w_feat=1.0,  # outer scaling is applied via w_ts_seg/w_ts_feat
            device=self.device,
        )

        # Infer feature strides once from a single cached case (workers read directly).
        self._cache_dir = Path(self._ts_cache_dir)
        self._feat_strides: Dict[str, Tuple[int, int, int]] = {}
        sample_labels = sorted(self._cache_dir.glob("*_ts_labels.b2nd"))
        if not sample_labels:
            raise FileNotFoundError(f"No *_ts_labels.b2nd files in {self._cache_dir}")
        n_cases = len(sample_labels)
        sample_case = sample_labels[0].name.replace("_ts_labels.b2nd", "")
        lbl = blosc2.open(urlpath=str(sample_labels[0]), mode="r")
        for stage in self.ts_feature_stages:
            fp = self._cache_dir / f"{sample_case}_ts_feat_{stage}.b2nd"
            if not fp.exists():
                raise FileNotFoundError(f"Missing cached feature: {fp}")
            f = blosc2.open(urlpath=str(fp), mode="r")
            self._feat_strides[stage] = tuple(int(round(lbl.shape[d] / f.shape[d + 1])) for d in range(3))

        self.print_to_log_file(
            f"TS structure loss active | cases cached: {n_cases} | "
            f"stages: {self.ts_feature_stages} | strides: {self._feat_strides} | "
            f"w_seg={self.w_ts_seg} w_feat={self.w_ts_feat}"
        )

    # ------------------------------------------------------------------
    # Data loader override: use the bbox-aware loader so train_step can crop
    # cached TS targets to the sampled patch.
    # ------------------------------------------------------------------

    def get_dataloaders(self):
        # Mirror the parent's setup but swap the loader class.
        if self.dataset_class is None:
            from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        patch = self.configuration_manager.patch_size
        kwargs = dict(
            label_manager=self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )
        _nnUNetDataLoaderWithBbox.set_ts_cache(
            self._cache_dir, self.ts_feature_stages, self._feat_strides
        )
        dl_tr = _nnUNetDataLoaderWithBbox(dataset_tr, self.batch_size, patch, patch, **kwargs)
        dl_val = _nnUNetDataLoaderWithBbox(dataset_val, self.batch_size, patch, patch, **kwargs)

        # Wrap in the same multithreaded augmenter the base trainer uses.
        from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
        from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
        from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
        allowed = get_allowed_n_proc_DA()
        if allowed == 0:
            return SingleThreadedAugmenter(dl_tr, None), SingleThreadedAugmenter(dl_val, None)
        return (
            NonDetMultiThreadedAugmenter(dl_tr, None, allowed, 1, None, pin_memory=self.device.type == "cuda"),
            NonDetMultiThreadedAugmenter(dl_val, None, max(1, allowed // 2), 1, None, pin_memory=self.device.type == "cuda"),
        )

    # ------------------------------------------------------------------
    # Helpers: gather cached targets for a batch
    # ------------------------------------------------------------------

    def _gather_cached_labels(
        self, keys: List[str], bboxes: List[List[Tuple[int, int]]], patch_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Return [B, 1, D, H, W] long tensor."""
        B = len(keys)
        out = np.zeros((B, 1, *patch_shape), dtype=np.int64)
        for b, (key, bbox) in enumerate(zip(keys, bboxes)):
            full = self._labels_cache[key]
            out[b, 0] = _crop_and_pad_3d(full, bbox, patch_shape, pad_value=0)
        return torch.from_numpy(out).to(self.device, non_blocking=True)

    def _gather_cached_features(
        self,
        keys: List[str],
        bboxes: List[List[Tuple[int, int]]],
        patch_shape: Tuple[int, ...],
    ) -> Dict[str, torch.Tensor]:
        """For each stage, return [B, C, d, h, w] aligned to the patch bbox."""
        result: Dict[str, torch.Tensor] = {}
        for stage in self.ts_feature_stages:
            stride = self._feat_strides[stage]
            feat_patch_shape = tuple(max(1, patch_shape[d] // stride[d]) for d in range(3))
            per_batch = []
            for key, bbox in zip(keys, bboxes):
                full = self._feats_cache[key][stage]  # (C, d, h, w)
                C = full.shape[0]
                feat_bbox = [
                    (bbox[d][0] // stride[d], bbox[d][0] // stride[d] + feat_patch_shape[d])
                    for d in range(3)
                ]
                # Read the (C, d, h, w) crop in one blosc2 slice
                slices = (slice(None),) + tuple(
                    slice(max(lo, 0), min(hi, full.shape[d + 1]))
                    for d, (lo, hi) in enumerate(feat_bbox)
                )
                crop = np.asarray(full[slices])
                # Pad if crop extended past volume bounds
                pads = [(0, 0)]
                for d, (lo, hi) in enumerate(feat_bbox):
                    lo_c = max(lo, 0)
                    hi_c = min(hi, full.shape[d + 1])
                    pads.append((lo_c - lo, (hi - lo) - (hi_c - lo)))
                if any(p != (0, 0) for p in pads):
                    crop = np.pad(crop, pads, mode="constant", constant_values=0.0)
                per_batch.append(crop)
            stacked = np.stack(per_batch, axis=0).astype(np.float32)
            result[stage] = torch.from_numpy(stacked).to(self.device, non_blocking=True)
        return result

    # ------------------------------------------------------------------
    # Train step
    # ------------------------------------------------------------------

    def train_step(self, batch: dict) -> dict:
        data = torch.from_numpy(batch["data"]).to(self.device, non_blocking=True)
        keys: List[str] = [str(k) for k in batch["keys"]]
        bboxes: List[List[Tuple[int, int]]] = batch["bbox"]  # supplied by subclass loader

        input_data = data[:, 0:1]
        target_data = data[:, 1:2]

        self.optimizer.zero_grad(set_to_none=True)
        ctx = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
        with ctx:
            output = self.network(input_data)

            if self.enable_deep_supervision:
                target_scales = self._downsample_target_for_ds(target_data)
                l_reg = self.loss(output, target_scales)
                pred_hr = output[0] if isinstance(output, (list, tuple)) else output
            else:
                l_reg = self.loss(output, target_data)
                pred_hr = output

        cached_labels = torch.from_numpy(batch["ts_labels"]).to(self.device, non_blocking=True)
        cached_feats = {
            stage: torch.from_numpy(arr).to(self.device, non_blocking=True)
            for stage, arr in batch["ts_feat"].items()
        }
        ts_out = self.ts_loss(pred_hr, cached_labels, cached_feats)

        l = l_reg + self.w_ts_seg * ts_out["seg_loss"] + self.w_ts_feat * ts_out["feat_loss"]

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {
            "loss": l.detach().cpu().numpy(),
            "loss_reg": l_reg.detach().cpu().numpy(),
            "loss_ts_seg": ts_out["seg_loss"].cpu().numpy(),
            "loss_ts_feat": ts_out["feat_loss"].cpu().numpy(),
        }

    # ------------------------------------------------------------------
    # Validation metrics: regression + segmentation Dice
    # ------------------------------------------------------------------

    def _compute_validation_metrics(self, validation_output_folder: str) -> None:
        """Regression metrics (MAE/PSNR/SSIM) then TS segmentation Dice."""
        super()._compute_validation_metrics(validation_output_folder)
        self._compute_segmentation_metrics(validation_output_folder)

    def _compute_segmentation_metrics(self, validation_output_folder: str) -> None:
        """
        Run TotalSegmentator on each saved sCT prediction and its paired GT CT,
        compute per-structure Dice for the body region defined by ``ts_eval_region``,
        and save a summary to ``segmentation_metrics.json``.

        GT CTs are expected at:
            {nnUNet_raw}/{dataset_name}/imagesTr/{case}_0001{file_ending}
        (modality index 1 = CT in the two-channel CBCT→CT setup).

        When ``TS_EVAL_CACHE_DIR`` is set (and pre-cached segmentations produced by
        ``scripts/cache_ts_targets.py --eval_gt_dir ... --eval_cache_dir ...`` are
        present as ``{case}_ts_fullseg*`` files), those are loaded directly instead
        of re-running TS on the GT, saving significant inference time.

        To add a new region: extend ``REGION_SEG_CONFIGS`` at the top of this
        module and set ``ts_eval_region`` on a subclass.
        """
        import SimpleITK as sitk
        from nnunetv2.paths import nnUNet_raw
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        region_structures = REGION_SEG_CONFIGS.get(self.ts_eval_region)
        if region_structures is None:
            self.print_to_log_file(
                f"No seg config for region '{self.ts_eval_region}', skipping seg metrics"
            )
            return

        ts_trainer_folder = _resolve_ts_trainer_folder(self._ts_results_dir, self._ts_config)
        ts_ds_json = load_json(os.path.join(ts_trainer_folder, "dataset.json"))
        label_name_to_int = {n: int(v) for n, v in ts_ds_json["labels"].items()}

        eval_structures = {s: label_name_to_int[s] for s in region_structures if s in label_name_to_int}
        missing = [s for s in region_structures if s not in label_name_to_int]
        if missing:
            self.print_to_log_file(f"TS label map missing (skipped): {missing}")
        if not eval_structures:
            self.print_to_log_file("No valid structures for seg eval, skipping")
            return

        file_ending = self.dataset_json["file_ending"]
        pred_folder = Path(validation_output_folder)
        gt_images_tr = Path(nnUNet_raw) / self.plans_manager.dataset_name / "imagesTr"

        case_pairs: List[Tuple[str, Path, Path]] = []
        for pred_file in sorted(pred_folder.glob(f"*{file_ending}")):
            k = pred_file.name[: -len(file_ending)]
            gt_file = gt_images_tr / f"{k}_0001{file_ending}"
            if gt_file.exists():
                case_pairs.append((k, pred_file, gt_file))
            else:
                self.print_to_log_file(f"GT CT not found for '{k}': {gt_file}")

        if not case_pairs:
            self.print_to_log_file(
                "No GT CT files found; skipping seg metrics", also_print_to_console=True
            )
            return

        self.print_to_log_file(
            f"TS seg eval: region='{self.ts_eval_region}', "
            f"{len(eval_structures)} structures, {len(case_pairs)} cases",
            also_print_to_console=True,
        )

        ts_predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=True,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        ts_predictor.initialize_from_trained_model_folder(
            ts_trainer_folder,
            use_folds=(self._ts_fold,),
            checkpoint_name="checkpoint_final.pth",
        )
        ts_file_ending = ts_predictor.dataset_json.get("file_ending", ".nii.gz")

        # Pre-cached GT segmentations (from cache_ts_targets.py --eval_cache_dir)
        eval_cache = Path(self._ts_eval_cache_dir) if self._ts_eval_cache_dir else None

        results: Dict[str, Dict[str, float]] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_lists: List[List[str]] = []
            output_truncated: List[str] = []
            case_roles: List[Tuple[str, str]] = []  # (case_key, "pred" | "gt")
            gt_seg_from_cache: Dict[str, Path] = {}  # k → pre-cached GT seg file

            for k, pred_file, gt_file in case_pairs:
                # Prediction always needs TS inference (changes every training run)
                input_lists.append([str(pred_file)])
                output_truncated.append(str(tmpdir / f"{k}_pred"))
                case_roles.append((k, "pred"))

                # GT: use pre-cached seg if available, otherwise infer with TS
                if eval_cache is not None:
                    cached = next(eval_cache.glob(f"{k}_ts_fullseg*"), None)
                    if cached is not None:
                        gt_seg_from_cache[k] = cached
                        continue

                input_lists.append([str(gt_file)])
                output_truncated.append(str(tmpdir / f"{k}_gt"))
                case_roles.append((k, "gt"))

            n_gt_cached = len(gt_seg_from_cache)
            n_gt_inferred = sum(1 for _, role in case_roles if role == "gt")
            self.print_to_log_file(
                f"  GT seg: {n_gt_cached} loaded from cache, {n_gt_inferred} via TS inference"
            )

            try:
                ts_predictor.predict_from_files(
                    input_lists,
                    output_truncated,
                    save_probabilities=False,
                    overwrite=True,
                    num_processes_preprocessing=2,
                    num_processes_segmentation_export=2,
                )
            except Exception as exc:
                self.print_to_log_file(
                    f"TS seg inference failed: {exc}", also_print_to_console=True
                )
                return

            seg_cache: Dict[str, np.ndarray] = {}
            for (k, role), trunc in zip(case_roles, output_truncated):
                seg_file = Path(trunc + ts_file_ending)
                if seg_file.exists():
                    seg_cache[f"{k}_{role}"] = sitk.GetArrayFromImage(
                        sitk.ReadImage(str(seg_file))
                    )
                else:
                    self.print_to_log_file(f"Missing TS output: {seg_file}")

            # Load pre-cached GT segmentations
            for k, cached_path in gt_seg_from_cache.items():
                seg_cache[f"{k}_gt"] = sitk.GetArrayFromImage(
                    sitk.ReadImage(str(cached_path))
                )

            for k, _, _ in case_pairs:
                pred_seg = seg_cache.get(f"{k}_pred")
                gt_seg = seg_cache.get(f"{k}_gt")
                if pred_seg is None or gt_seg is None:
                    continue

                case_dice: Dict[str, float] = {}
                for struct_name, label_id in eval_structures.items():
                    pm = pred_seg == label_id
                    gm = gt_seg == label_id
                    denom = int(pm.sum()) + int(gm.sum())
                    case_dice[struct_name] = (
                        float(2 * int((pm & gm).sum()) / denom) if denom > 0 else 1.0
                    )
                results[k] = case_dice
                self.print_to_log_file(
                    f"  {k}: mean Dice = {np.mean(list(case_dice.values())):.3f}"
                )

        if not results:
            return

        struct_dice_lists: Dict[str, List[float]] = {s: [] for s in eval_structures}
        for case_dice in results.values():
            for s, d in case_dice.items():
                struct_dice_lists[s].append(d)
        mean_per_struct = {s: float(np.mean(v)) for s, v in struct_dice_lists.items() if v}
        overall_mean = (
            float(np.mean(list(mean_per_struct.values()))) if mean_per_struct else float("nan")
        )

        self.print_to_log_file(
            f"=== TS Segmentation Dice ({self.ts_eval_region}) ===", also_print_to_console=True
        )
        for struct_name in sorted(mean_per_struct):
            self.print_to_log_file(
                f"  {struct_name}: {mean_per_struct[struct_name]:.3f}", also_print_to_console=True
            )
        self.print_to_log_file(f"  Mean Dice: {overall_mean:.3f}", also_print_to_console=True)
        self.print_to_log_file("=" * 45, also_print_to_console=True)

        summary = {
            "region": self.ts_eval_region,
            "mean_dice": overall_mean,
            "structures": mean_per_struct,
            "per_case": results,
        }
        summary_file = os.path.join(validation_output_folder, "segmentation_metrics.json")
        with open(summary_file, "w") as fh:
            _json.dump(summary, fh, indent=2)
        self.print_to_log_file(
            f"Segmentation metrics saved to: {summary_file}", also_print_to_console=True
        )


# ---------------------------------------------------------------------------
# Bbox utilities
# ---------------------------------------------------------------------------

def _crop_and_pad_3d(
    arr: np.ndarray,
    bbox: List[Tuple[int, int]],
    patch_shape: Tuple[int, ...],
    pad_value=0,
) -> np.ndarray:
    """
    Crop ``arr`` (D, H, W) to the given bbox and zero-pad to ``patch_shape``.

    Handles out-of-bounds bbox extents (can happen because nnUNet's random
    patch sampler deliberately drifts outside the volume; those voxels are
    padded with ``pad_value``).
    """
    shape = arr.shape
    dim = len(shape)
    slices, pads = [], []
    for d in range(dim):
        lo, hi = bbox[d]
        lo_c = max(lo, 0)
        hi_c = min(hi, shape[d])
        slices.append(slice(lo_c, hi_c))
        pads.append((lo_c - lo, (hi - lo) - (hi_c - lo)))
    cropped = np.asarray(arr[tuple(slices)])
    if any(p != (0, 0) for p in pads):
        cropped = np.pad(cropped, pads, mode="constant", constant_values=pad_value)
    # Patch shape may differ from (hi - lo) due to padding asymmetry; final slice/pad to match.
    out = np.full(patch_shape, pad_value, dtype=cropped.dtype)
    final_slice = tuple(slice(0, min(patch_shape[d], cropped.shape[d])) for d in range(dim))
    out[final_slice] = cropped[final_slice]
    return out


