"""
Trace and save a MedDiNOv3 feature extractor for use with MedicalPerceptualLoss.

This script contains the MedDiNOv3 checkpoint conversion and loading logic
directly so it does not depend on any external helper file. It traces a wrapper
that:
1. loads the Hugging Face-style MedDiNOv3 backbone,
2. captures one or more intermediate layer activations with forward hooks,
3. pools each captured feature to a per-sample embedding,
4. returns either a single feature tensor or a tuple of feature tensors.

The traced model is intended to be consumed by
``nnunetv2.training.loss.regression_losses.MedicalPerceptualLoss``.

Usage
-----
    # Trace final pooled output only
    python scripts/trace_perceptual_model.py \
        --output perceptual_model.pt

    # Trace selected intermediate layers
    python scripts/trace_perceptual_model.py \
        --checkpoint /path/to/model.pth \
        --feature-layers layer.3 layer.7 layer.11 \
        --output perceptual_model.pt

Arguments
---------
    --checkpoint     : Optional MedDiNOv3 checkpoint path. If omitted, the
                       default Hugging Face Hub checkpoint is used.
    --hf-repo        : Hugging Face repo id used when ``--checkpoint`` is not
                       provided.
    --feature-layers : One or more module names to capture with forward hooks.
                       Names must match ``model.named_modules()`` of the loaded
                       ``DINOv3ViTModel``.
    --input-size     : Spatial size (H=W) of the square input slice in pixels.
                       Default: 224.
    --output         : Path to save the traced .pt model.
                       Default: perceptual_model.pt.
    --device         : 'cpu' or 'cuda'. Default: 'cuda' if available, else 'cpu'.

Notes
-----
- The traced model accepts a ``(B, 3, H, W)`` float32 tensor.
- If ``--feature-layers`` is omitted, the wrapper returns the model
  ``pooler_output``.
- If ``--feature-layers`` is provided, the wrapper returns pooled intermediate
  features in the requested order.
- For feature tensors with spatial or token dimensions, mean pooling is applied
  over all non-batch, non-channel dimensions.
"""

import argparse
from typing import Dict, List, Optional

import torch
from huggingface_hub import hf_hub_download


def convert_state_dict(state_dict: dict) -> dict:
    """Map DINOv3/MedDiNOv3 checkpoint keys to Transformers key names."""
    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        if key.startswith("patch_embed.proj"):
            new_key = key.replace("patch_embed.proj", "embeddings.patch_embeddings")
        elif key.startswith("rope_embed.periods"):
            new_key = key.replace("rope_embed.periods", "rope_embeddings.inv_freq")
            value = 1.0 / value
            if value.dtype != torch.float32:
                value = value.to(torch.float32)
        elif key == "cls_token":
            new_key = "embeddings.cls_token"
        elif key in ("register_tokens", "storage_tokens"):
            new_key = "embeddings.register_tokens"
        elif key == "mask_token":
            new_key = "embeddings.mask_token"
        elif key.startswith("blocks."):
            parts = key.split(".")
            layer_idx = parts[1]
            rest = ".".join(parts[2:])

            if rest.startswith("attn.qkv"):
                suffix = rest.replace("attn.qkv", "")
                if suffix == ".weight":
                    q, k, v = value.chunk(3, dim=0)
                    new_state_dict[f"layer.{layer_idx}.attention.q_proj.weight"] = q
                    new_state_dict[f"layer.{layer_idx}.attention.k_proj.weight"] = k
                    new_state_dict[f"layer.{layer_idx}.attention.v_proj.weight"] = v
                    continue
                if suffix == ".bias":
                    q, k, v = value.chunk(3, dim=0)
                    new_state_dict[f"layer.{layer_idx}.attention.q_proj.bias"] = q
                    new_state_dict[f"layer.{layer_idx}.attention.k_proj.bias"] = k
                    new_state_dict[f"layer.{layer_idx}.attention.v_proj.bias"] = v
                    continue
            elif rest.startswith("attn.proj"):
                new_key = f"layer.{layer_idx}.attention.o_proj{rest.replace('attn.proj', '')}"
            elif rest.startswith("norm1"):
                new_key = f"layer.{layer_idx}.norm1{rest.replace('norm1', '')}"
            elif rest.startswith("norm2"):
                new_key = f"layer.{layer_idx}.norm2{rest.replace('norm2', '')}"
            elif rest == "ls1.gamma":
                new_key = f"layer.{layer_idx}.layer_scale1.lambda1"
            elif rest == "ls2.gamma":
                new_key = f"layer.{layer_idx}.layer_scale2.lambda1"
            elif rest.startswith("mlp.fc1"):
                new_key = f"layer.{layer_idx}.mlp.up_proj{rest.replace('mlp.fc1', '')}"
            elif rest.startswith("mlp.fc2"):
                new_key = f"layer.{layer_idx}.mlp.down_proj{rest.replace('mlp.fc2', '')}"
            else:
                new_key = f"layer.{layer_idx}.{rest}"

        if new_key == "embeddings.mask_token":
            value = value.unsqueeze(0)

        new_state_dict[new_key] = value

    return new_state_dict


def load_meddinov3_transformers(
    checkpoint_path: Optional[str] = None,
    hf_repo: str = "ricklisz123/MedDINOv3-ViTB-16-CT-3M",
    device: str = "cpu",
):
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    config = DINOv3ViTConfig(
        patch_size=16,
        hidden_size=768,
        intermediate_size=768 * 4,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_register_tokens=4,
        query_bias=False,
        key_bias=False,
        value_bias=False,
        proj_bias=True,
        mlp_bias=True,
        use_gated_mlp=False,
        layerscale_value=1e-5,
        drop_path_rate=0.2,
        image_size=224,
        rope_theta=100.0,
    )

    model = DINOv3ViTModel(config)

    if checkpoint_path is None:
        checkpoint_path = hf_hub_download(repo_id=hf_repo, filename="model.pth")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "teacher" in checkpoint:
        state_dict = checkpoint["teacher"]
        state_dict = {
            k.replace("backbone.", ""): v
            for k, v in state_dict.items()
            if "ibot" not in k and "dino_head" not in k
        }
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    converted_state_dict = convert_state_dict(state_dict)
    model.load_state_dict(converted_state_dict, strict=False)
    model.rope_embeddings.inv_freq = converted_state_dict["rope_embeddings.inv_freq"]
    model = model.to(device)
    model.eval()
    return model


class _MedDiNOv3FeatureWrapper(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, feature_layers: Optional[List[str]] = None):
        super().__init__()
        self.backbone = backbone
        self.feature_layers = list(feature_layers) if feature_layers is not None else []
        self._features: Dict[str, torch.Tensor] = {}

        self._handles = []
        if len(self.feature_layers) > 0:
            named_modules = dict(self.backbone.named_modules())
            missing = [name for name in self.feature_layers if name not in named_modules]
            if len(missing) > 0:
                available = list(named_modules.keys())
                preview = ", ".join(available[:50])
                raise ValueError(
                    "The following feature layers were not found in the model: "
                    f"{missing}. Available module names include: {preview}"
                )

            for name in self.feature_layers:
                self._handles.append(named_modules[name].register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, (list, tuple)):
                tensor_outputs = [o for o in output if isinstance(o, torch.Tensor)]
                if len(tensor_outputs) == 0:
                    raise TypeError(f"Layer '{name}' produced no tensor output.")
                output = tensor_outputs[0]
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Layer '{name}' produced unsupported output type: {type(output)}")
            self._features[name] = output
        return hook

    @staticmethod
    def _pool_feature(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x
        if x.ndim == 3:
            return x.mean(dim=1)
        if x.ndim >= 4:
            dims = tuple(range(2, x.ndim))
            return x.mean(dim=dims)
        raise ValueError(f"Unsupported feature tensor shape: {tuple(x.shape)}")

    def forward(self, x: torch.Tensor):
        self._features = {}
        outputs = self.backbone(x)

        if len(self.feature_layers) == 0:
            if not hasattr(outputs, "pooler_output"):
                raise RuntimeError("MedDiNOv3 backbone output does not expose pooler_output.")
            return outputs.pooler_output

        collected = []
        for name in self.feature_layers:
            if name not in self._features:
                raise RuntimeError(f"Requested feature layer '{name}' did not produce an activation.")
            collected.append(self._pool_feature(self._features[name]))

        if len(collected) == 1:
            return collected[0]

        return tuple(collected)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional MedDiNOv3 checkpoint path.",
    )
    parser.add_argument(
        "--hf-repo",
        default="ricklisz123/MedDINOv3-ViTB-16-CT-3M",
        help="Hugging Face repo id used when checkpoint is not provided.",
    )
    parser.add_argument(
        "--feature-layers",
        nargs="+",
        default=None,
        help="One or more module names to capture with forward hooks.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Spatial size (H=W) of the square input slice in pixels.",
    )
    parser.add_argument(
        "--output",
        default="perceptual_model.pt",
        help="Output path for the traced .pt model.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for tracing.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading MedDiNOv3 backbone ...")
    backbone = load_meddinov3_transformers(
        checkpoint_path=args.checkpoint,
        hf_repo=args.hf_repo,
        device=str(device),
    )

    if not isinstance(backbone, torch.nn.Module):
        raise TypeError("load_meddinov3_transformers did not return a torch.nn.Module.")

    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    model = _MedDiNOv3FeatureWrapper(backbone, args.feature_layers).to(device)
    model.eval()

    dummy = torch.zeros(1, 3, args.input_size, args.input_size, dtype=torch.float32, device=device)

    print(f"Tracing model with input shape {tuple(dummy.shape)} ...")
    if args.feature_layers is None:
        print("Capturing final pooled MedDiNOv3 output.")
    else:
        print(f"Capturing layers: {args.feature_layers}")

    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)
        out = traced(dummy)

    if isinstance(out, torch.Tensor):
        print(f"  Output shape: {tuple(out.shape)}")
    elif isinstance(out, (tuple, list)):
        print(f"  Number of outputs: {len(out)}")
        for idx, tensor in enumerate(out):
            print(f"    [{idx}] shape: {tuple(tensor.shape)}")
    else:
        raise TypeError(f"Unexpected traced output type: {type(out)}")

    traced.save(args.output)
    print(f"Traced model saved to: {args.output}")
    print()
    print("Pass this path to RegressionCompoundLoss or MedicalPerceptualLoss:")
    print(f"  RegressionCompoundLoss(w_perc=0.05, perceptual_model_path='{args.output}')")


if __name__ == "__main__":
    main()
