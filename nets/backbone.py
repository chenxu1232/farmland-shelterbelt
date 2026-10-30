
import math
import warnings
import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Helper Functions
# ==============================================================================
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

# ==============================================================================
# 1. NEW: Channel Attention Adapter (Core module for handling multiband inputs)
# ==============================================================================
class ChannelAttentionAdapter(nn.Module):
    """
    Adapt multiband data (e.g., 8 channels) to the 3-channel input of SegFormer,
    and automatically learn band importance using SE attention.
    """
    def __init__(self, in_chans, out_chans=3, reduction=4):
        super().__init__()
        # Channel attention (SE module)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_chans, in_chans // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_chans // reduction, in_chans, bias=False),
            nn.Sigmoid()
        )
        # Channel reduction: N -> 3
        self.mapping = nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False)
        
        # Initialization
        self._init_weights(in_chans, out_chans)

    def _init_weights(self, in_chans, out_chans):
        # 1. Initialize the mapping layer
        # Ideally, the initialization should retain as much original RGB information as possible (if the first three channels are RGB)
        nn.init.kaiming_normal_(self.mapping.weight, mode='fan_out', nonlinearity='relu')
        
        # If the first three channels are known to be RGB, explicit initialization is an option (standard Kaiming initialization is used here)

    def forward(self, x):
        b, c, _, _ = x.size()
        # 1. SE attention: learn band weights
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x) # Dynamic weighting
        
        # 2. Mapping: project to 3 channels so the downstream network can load pretrained weights
        x = self.mapping(x)
        return x

# ==============================================================================
# Existing Modules (OverlapPatchEmbed, Attention, etc.)
# ==============================================================================

class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

# ===================== FSA: DCT-based Frequency Self-Attention =====================
class _DCTCache:
    def __init__(self):
        self.cache = {}

    @staticmethod
    def _dct_orthonormal(n, device, dtype):
        k = torch.arange(n, device=device, dtype=dtype).view(1, -1)
        x = torch.arange(n, device=device, dtype=dtype).view(-1, 1)
        mat = torch.cos(math.pi * (x + 0.5) * k / n)
        mat[:, 0] = mat[:, 0] / math.sqrt(n)
        mat[:, 1:] = mat[:, 1:] * math.sqrt(2.0 / n)
        return mat

    def get_operators(self, H, W, kq, kv, device, dtype):
        key = (H, W, kq, kv, device, dtype)
        if key in self.cache:
            return self.cache[key]
        D_H = self._dct_orthonormal(H, device, dtype)
        D_W = self._dct_orthonormal(W, device, dtype)
        kq = min(kq, H, W)
        kv = min(kv, H, W)
        PqH, PqW = D_H[:kq, :], D_W[:kq, :]
        GqH, GqW = D_H[:, :kq], D_W[:, :kq]
        PkH, PkW = D_H[:kv, :], D_W[:kv, :]
        GkH, GkW = D_H[:, :kv], D_W[:, :kv]
        self.cache[key] = (PqH, PqW, GqH, GqW, PkH, PkW, GkH, GkW, kq, kv)
        return self.cache[key]

_DCT = _DCTCache()

class FSA_DCT_Attn_Token(nn.Module):
    def __init__(self, dim, num_heads=4, kq=16, kv=8, qkv_bias=True, use_linear_attn=False, eps=1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.kq = kq
        self.kv = kv
        self.use_linear = use_linear_attn
        self.eps = eps
        
        self.q_proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.k_proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.v_proj = nn.Conv2d(dim, dim, 1, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        
        # FSA frequency-domain channel attention branch
        self.freq_pool = nn.AdaptiveAvgPool2d(1)
        self.freq_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.freq_act  = nn.Sigmoid()
        self.last_freq_map = None

    @staticmethod
    def _to_heads_2d(t, heads):
        B, C, h, w = t.shape
        d = C // heads
        return t.view(B, heads, d, h, w).flatten(-2).permute(0, 1, 3, 2)

    @staticmethod
    def _from_heads_2d(t, heads, C, h, w):
        B, _, N, d = t.shape
        return t.permute(0,1,3,2).contiguous().view(B, C, h, w)

    def forward(self, x, H, W):
        B, N, C = x.shape
        assert N == H * W
        x_sp = x.transpose(1, 2).reshape(B, C, H, W)
        
        PqH, PqW, GqH, GqW, PkH, PkW, GkH, GkW, kq, kv = _DCT.get_operators(
            H, W, self.kq, self.kv, x.device, x.dtype
        )

        q_low = torch.einsum('bchw,kh,lw->bckl', x_sp, PqH, PqW)
        k_low = torch.einsum('bchw,kh,lw->bckl', x_sp, PkH, PkW)
        v_low = torch.einsum('bchw,kh,lw->bckl', x_sp, PkH, PkW)

        # Freq Map Calculation
        freq = self.freq_pool(q_low)
        freq = self.freq_conv(freq)
        freq = self.freq_act(freq)
        self.last_freq_map = freq

        q_map = self.q_proj(q_low)
        k_map = self.k_proj(k_low)
        v_map = self.v_proj(v_low)

        q = self._to_heads_2d(q_map, self.num_heads)
        k = self._to_heads_2d(k_map, self.num_heads)
        v = self._to_heads_2d(v_map, self.num_heads)

        if not self.use_linear:
            att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            att = att.softmax(dim=-1)
            out_low = torch.matmul(att, v)
        else:
            phi = lambda t: F.elu(t, alpha=1.0, inplace=False) + 1.0
            qf, kf = phi(q), phi(k)
            kv_ = torch.matmul(kf.transpose(-2, -1), v)
            z_  = torch.sum(kf, dim=-2)
            out_num = torch.matmul(qf, kv_)
            out_den = torch.matmul(qf, z_.unsqueeze(-1)).clamp_min(1e-6)
            out_low = out_num / out_den

        out_low_2d = self._from_heads_2d(out_low, self.num_heads, C, kq, kq)
        out_sp = torch.einsum('bckl,hk,wl->bchw', out_low_2d, GqH, GqW)
        
        out = out_sp.flatten(2).transpose(1, 2).contiguous()
        out = self.out_proj(out)
        return out

# ===================== Standard Attention & Blocks =====================
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=GELU, 
                 norm_layer=nn.LayerNorm, sr_ratio=1,
                 use_fsa=False, fsa_kq=16, fsa_kv=8, fsa_linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if use_fsa:
            self.attn = FSA_DCT_Attn_Token(dim, num_heads, fsa_kq, fsa_kv, qkv_bias, fsa_linear)
        else:
            self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)
        
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

# ==============================================================================
# Main Backbone Class: MixVisionTransformer
# ==============================================================================
class MixVisionTransformer(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, embed_dims=[32, 64, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # ====================================================
        # NEW: Channel Adapter Logic
        # Enable the adapter when the input has a non-3-channel format (e.g., 8 channels)
        # Pass through 3-channel inputs directly, without adding parameters
        # ====================================================
        if in_chans != 3:
            self.channel_adapter = ChannelAttentionAdapter(in_chans, out_chans=3)
        else:
            self.channel_adapter = nn.Identity()

        # Transformer parts
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Block 1
        # Note: in_chans is fixed at 3 here because the preceding adapter has already converted N channels to 3
        # This preserves compatibility with the pretrained weights
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=3, embed_dim=embed_dims[0])
        cur = 0
        self.block1 = nn.ModuleList([
            Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, 
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], 
                  norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])

        # Block 2
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, 
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], 
                  norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])

        # Block 3
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, 
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], 
                  norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])

        # Block 4 (With FSA)
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, 
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], 
                  norm_layer=norm_layer, sr_ratio=sr_ratios[3],
                  # Enable the FSA module
                  use_fsa=True, fsa_kq=16, fsa_kv=8, fsa_linear=False)
            for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # NEW: First process multiband inputs through the Channel Adapter
        # x shape: [B, in_chans, H, W] -> [B, 3, H, W]
        x = self.channel_adapter(x)

        B = x.shape[0]
        outs = []

        # Stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # Stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        
        # Extract the FSA freq_map
        freq_map4 = None
        last_attn = self.block4[-1].attn
        if isinstance(last_attn, FSA_DCT_Attn_Token) and last_attn.last_freq_map is not None:
            freq_map4 = last_attn.last_freq_map

        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs, freq_map4

# ==============================================================================
# Backbone Variants (B0 - B5)
# ==============================================================================
# Note: **kwargs has been added to every class so that in_chans can be passed in

class mit_b0(MixVisionTransformer):
    def __init__(self, pretrained=False, **kwargs):
        super(mit_b0, self).__init__(
            embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)
        if pretrained:
            print("Load backbone weights")
            self.load_state_dict(torch.load("model_data/segformer_b0_backbone_weights.pth"), strict=False)

class mit_b1(MixVisionTransformer):
    def __init__(self, pretrained=False, **kwargs):
        super(mit_b1, self).__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)
        if pretrained:
            print("Load backbone weights")
            self.load_state_dict(torch.load("model_data/segformer_b1_backbone_weights.pth"), strict=False)

class mit_b2(MixVisionTransformer):
    def __init__(self, pretrained=False, **kwargs):
        super(mit_b2, self).__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)
        if pretrained:
            print("Load backbone weights")
            self.load_state_dict(torch.load("model_data/segformer_b2_backbone_weights.pth"), strict=False)

class mit_b3(MixVisionTransformer):
    def __init__(self, pretrained=False, **kwargs):
        super(mit_b3, self).__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)
        if pretrained:
            print("Load backbone weights")
            # With the adapter, the first backbone layer still accepts 3 channels, so weights can be loaded directly without the previous complex filtering logic
            # As a precaution against unrelated keys in the .pth file, use strict=False
            self.load_state_dict(torch.load("model_data/segformer_b3_backbone_weights.pth"), strict=False)

class mit_b4(MixVisionTransformer):
    def __init__(self, pretrained=False, **kwargs):
        super(mit_b4, self).__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)
        if pretrained:
            print("Load backbone weights")
            self.load_state_dict(torch.load("model_data/segformer_b4_backbone_weights.pth"), strict=False)

class mit_b5(MixVisionTransformer):
    def __init__(self, pretrained=False, **kwargs):
        super(mit_b5, self).__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)
        if pretrained:
            print("Load backbone weights")
            # Similarly, simplify the loading logic because the architectures now match
            self.load_state_dict(torch.load("model_data/segformer_b5_backbone_weights.pth"), strict=False)
