"""
Attention-augmented UNet components for 3D image-to-image translation (e.g., CBCT→CT).

Components:
- AttentionGate3D: Oktay-style additive attention gates on skip connections
- SelfAttentionBottleneck3D: Multi-head self-attention at the UNet bottleneck
- UNetDecoder_Trilinear_AttentionGates: Trilinear upsampling decoder with optional attention
- PlainConvUNetRegression: PlainConvUNet with attention decoder for regression
- ResidualEncoderUNetRegression: ResidualEncoderUNet with attention decoder for regression

Reference: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas", MIDL 2018
"""

from typing import Union, List, Tuple, Type

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.unet_decoder_upsample_trilinear import UpsampleAndConv3D
from dynamic_network_architectures.initialization.weight_init import InitWeights_He, init_last_bn_before_add_to_0


class AttentionGate3D(nn.Module):
    """
    Oktay-style additive attention gate for 3D UNet skip connections.

    Computes a soft spatial attention mask using the upsampled gating signal (g)
    from the decoder and the encoder skip connection (x). The mask suppresses
    source-modality artifacts on the skip path.

    Args:
        F_g: Channels in the gating signal (upsampled decoder features).
        F_l: Channels in the skip connection (encoder features).
        F_int: Intermediate channels for the attention computation (typically F_l // 2).
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Conv3d(F_g, F_int, kernel_size=1, bias=True)
        self.W_x = nn.Conv3d(F_l, F_int, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

        # Initialize psi bias negative so gates start mostly open (avoids dead-start)
        nn.init.constant_(self.psi[0].bias, -0.1)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal — upsampled decoder features [B, F_g, D, H, W].
            x: Skip connection — encoder features at same resolution [B, F_l, D, H, W].

        Returns:
            Attention-gated skip features [B, F_l, D, H, W].
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # [B, 1, D, H, W]
        return x * psi


class SelfAttentionBottleneck3D(nn.Module):
    """
    Multi-head self-attention (MHSA) applied at the UNet bottleneck.

    Flattens the 3D spatial dimensions into a sequence of tokens, applies
    MHSA with a residual connection, then reshapes back. Memory cost is
    O(N^2 * C) where N = D*H*W at the bottleneck — typically 8–64 tokens
    at nnUNet's default bottleneck resolution.

    Args:
        channels: Number of feature channels at the bottleneck.
        num_heads: Number of attention heads (adjusted down if channels not divisible).
        dropout: Attention dropout probability.
    """

    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        # Halve num_heads until channels is divisible
        while num_heads > 1 and channels % num_heads != 0:
            num_heads //= 2
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, *spatial = x.shape
        N = int(np.prod(spatial))
        x_flat = x.view(B, C, N).permute(0, 2, 1)  # (B, N, C)
        x_norm = self.norm(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out  # residual
        return x_flat.permute(0, 2, 1).view(B, C, *spatial)


class UNetDecoder_Trilinear_AttentionGates(nn.Module):
    """
    UNet decoder using trilinear+conv upsampling with optional Oktay attention gates
    on skip connections and optional self-attention at the bottleneck.

    Builds on ``UpsampleAndConv3D`` (axis-wise trilinear interp + 3×3×3 conv) from
    the dynamic_network_architectures package and adds:
    - Per-stage ``AttentionGate3D`` to gate encoder skips before concatenation.
    - A single ``SelfAttentionBottleneck3D`` applied to the bottleneck features
      before the first decode stage.

    Args:
        encoder: Encoder module (PlainConvEncoder or ResidualEncoder).
        num_classes: Number of output channels.
        n_conv_per_stage: Conv blocks per decoder stage.
        deep_supervision: If True, produce a prediction at every resolution level.
        use_attention_gates: Whether to apply attention gates on skip connections.
        use_bottleneck_sa: Whether to apply self-attention at the bottleneck.
        bottleneck_sa_heads: Number of MHSA heads for the bottleneck SA module.
        nonlin_first: Conv → nonlin → norm if True; else conv → norm → nonlin.
        norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
        conv_bias: Fall back to encoder values if None.
    """

    def __init__(
        self,
        encoder: Union[PlainConvEncoder, ResidualEncoder],
        num_classes: int,
        n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
        deep_supervision: bool,
        use_attention_gates: bool = True,
        use_bottleneck_sa: bool = False,
        bottleneck_sa_heads: int = 8,
        nonlin_first: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        conv_bias: bool = None,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.use_attention_gates = use_attention_gates
        self.encoder = encoder
        self.num_classes = num_classes

        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, (
            "n_conv_per_stage must have as many entries as we have resolution stages - 1 "
            f"(n_stages in encoder - 1), here: {n_stages_encoder}"
        )

        # Fall back to encoder settings
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs

        stages = []
        transpconvs = []
        seg_layers = []
        attention_gates = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]

            transpconvs.append(
                UpsampleAndConv3D(input_features_below, input_features_skip, stride_for_transpconv, conv_bias)
            )

            if use_attention_gates:
                F_int = max(input_features_skip // 2, 1)
                attention_gates.append(AttentionGate3D(input_features_skip, input_features_skip, F_int))

            stages.append(
                StackedConvBlocks(
                    n_conv_per_stage[s - 1],
                    encoder.conv_op,
                    2 * input_features_skip,
                    input_features_skip,
                    encoder.kernel_sizes[-(s + 1)],
                    1,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first,
                )
            )

            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.attention_gates = nn.ModuleList(attention_gates) if use_attention_gates else None

        # Bottleneck self-attention (applied to skips[-1] before first decode stage)
        if use_bottleneck_sa:
            bottleneck_channels = encoder.output_channels[-1]
            self.bottleneck_sa = SelfAttentionBottleneck3D(bottleneck_channels, bottleneck_sa_heads)
        else:
            self.bottleneck_sa = None

    def forward(self, skips: List[torch.Tensor]):
        """
        Args:
            skips: List of encoder feature maps, ordered by resolution
                   (skips[-1] is the bottleneck / lowest resolution).

        Returns:
            If deep_supervision: list of predictions at each scale, largest first.
            Else: single prediction at full resolution.
        """
        lres_input = skips[-1]

        if self.bottleneck_sa is not None:
            lres_input = self.bottleneck_sa(lres_input)

        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            skip = skips[-(s + 2)]
            if self.attention_gates is not None:
                skip = self.attention_gates[s](x, skip)
            x = torch.cat((x, skip), dim=1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == len(self.stages) - 1:
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # Return largest resolution first
        seg_outputs = seg_outputs[::-1]
        return seg_outputs if self.deep_supervision else seg_outputs[0]

    def compute_conv_feature_map_size(self, input_size):
        """
        Estimate intermediate feature map memory (in elements) for a given input size.
        ``input_size`` is the spatial size fed to the *encoder* (no batch/channel dims).
        """
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]

        assert len(skip_sizes) == len(self.stages)

        output = np.int64(0)
        for s in range(len(self.stages)):
            spatial = skip_sizes[-(s + 1)]
            F_skip = self.encoder.output_channels[-(s + 2)]

            # Conv blocks
            output += self.stages[s].compute_conv_feature_map_size(spatial)
            # Transpconv output
            output += np.prod([F_skip, *spatial], dtype=np.int64)
            # Seg layer
            if self.deep_supervision or s == len(self.stages) - 1:
                output += np.prod([self.num_classes, *spatial], dtype=np.int64)
            # Attention gate intermediates: W_g + W_x + psi outputs
            if self.use_attention_gates:
                F_int = max(F_skip // 2, 1)
                output += np.int64(3) * F_int * np.prod(spatial, dtype=np.int64)

        return output


class PlainConvUNetRegression(nn.Module):
    """
    Plain-conv UNet for 3D image regression (e.g., CBCT→CT) with:
    - Trilinear + 3×3×3 conv upsampling (STU-Net style, no nearest+1×1 artifacts)
    - Optional Oktay attention gates on every skip connection
    - Optional multi-head self-attention at the bottleneck

    Designed to be instantiated by nnUNet's ``get_network_from_plans`` via
    ``pydoc.locate("nnunetv2.network_architecture.attention_unet.PlainConvUNetRegression")``.

    Extra arch_kwargs compared to PlainConvUNet:
        use_attention_gates (bool): Enable attention gates. Default True.
        use_bottleneck_sa (bool): Enable bottleneck self-attention. Default False.
        bottleneck_sa_heads (int): MHSA heads for bottleneck SA. Default 8.
    """

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...], None] = None,
        num_classes: int = 1,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]] = 2,
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...], None] = None,
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
        use_attention_gates: bool = True,
        use_bottleneck_sa: bool = False,
        bottleneck_sa_heads: int = 8,
    ):
        super().__init__()

        # Accept n_blocks_per_stage as alias (used by ResEnc plans)
        if n_conv_per_stage is None and n_blocks_per_stage is not None:
            n_conv_per_stage = n_blocks_per_stage
        assert n_conv_per_stage is not None, "Either n_conv_per_stage or n_blocks_per_stage must be provided"

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        assert len(n_conv_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == n_stages - 1

        self.encoder = PlainConvEncoder(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs, return_skips=True, nonlin_first=nonlin_first,
        )
        self.decoder = UNetDecoder_Trilinear_AttentionGates(
            self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
            use_attention_gates=use_attention_gates,
            use_bottleneck_sa=use_bottleneck_sa,
            bottleneck_sa_heads=bottleneck_sa_heads,
            nonlin_first=nonlin_first,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "Give input_size=(x, y, z) without batch/channel dimensions."
        )
        return (
            self.encoder.compute_conv_feature_map_size(input_size)
            + self.decoder.compute_conv_feature_map_size(input_size)
        )

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


class ResidualEncoderUNetRegression(nn.Module):
    """
    Residual-encoder UNet for 3D image regression (e.g., CBCT→CT) with:
    - Trilinear + 3×3×3 conv upsampling in the decoder
    - Optional Oktay attention gates on every skip connection
    - Optional multi-head self-attention at the bottleneck

    Designed to be instantiated by nnUNet's ``get_network_from_plans`` via
    ``pydoc.locate("nnunetv2.network_architecture.attention_unet.ResidualEncoderUNetRegression")``.

    Extra arch_kwargs compared to ResidualEncoderUNet:
        use_attention_gates (bool): Enable attention gates. Default True.
        use_bottleneck_sa (bool): Enable bottleneck self-attention. Default False.
        bottleneck_sa_heads (int): MHSA heads for bottleneck SA. Default 8.
    """

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
        bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
        stem_channels: int = None,
        use_attention_gates: bool = True,
        use_bottleneck_sa: bool = False,
        bottleneck_sa_heads: int = 8,
    ):
        super().__init__()

        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        assert len(n_blocks_per_stage) == n_stages
        assert len(n_conv_per_stage_decoder) == n_stages - 1

        self.encoder = ResidualEncoder(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
            n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs, block, bottleneck_channels,
            return_skips=True, disable_default_stem=False, stem_channels=stem_channels,
        )
        self.decoder = UNetDecoder_Trilinear_AttentionGates(
            self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
            use_attention_gates=use_attention_gates,
            use_bottleneck_sa=use_bottleneck_sa,
            bottleneck_sa_heads=bottleneck_sa_heads,
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "Give input_size=(x, y, z) without batch/channel dimensions."
        )
        return (
            self.encoder.compute_conv_feature_map_size(input_size)
            + self.decoder.compute_conv_feature_map_size(input_size)
        )

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)
