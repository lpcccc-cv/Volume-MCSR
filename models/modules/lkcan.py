import torch
import torch.nn as nn


# -------------------------
# Depthwise Convolution
# -------------------------
class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x):
        return self.dwconv(x)


# -------------------------
# Feed-Forward Net
# -------------------------
class FFN(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.dwconv_1 = nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1)
        self.act = nn.ReLU()
        self.dwconv_2 = nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.dwconv_2(self.act(self.dwconv_1(x)))
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.conv0_1 = nn.Conv2d(dim, dim, kernel_size=(1, 7), padding=(0, 3), groups=1)
        self.conv0_2 = nn.Conv2d(dim, dim, kernel_size=(7, 1), padding=(3, 0), groups=1)
        self.conv1_1 = nn.Conv2d(dim, dim, kernel_size=(1, 11), padding=(0, 5), groups=1)
        self.conv1_2 = nn.Conv2d(dim, dim, kernel_size=(11, 1), padding=(5, 0), groups=1)
        self.conv2_1 = nn.Conv2d(dim, dim, kernel_size=(1, 15), padding=(0, 7), groups=1)
        self.conv2_2 = nn.Conv2d(dim, dim, kernel_size=(15, 1), padding=(7, 0), groups=1)
        self.relu = nn.ReLU()
        self.lkc_fusion = nn.Conv2d(dim*3, dim, 3,1,1)

        self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.conv_att_1 = nn.Conv2d(dim, dim, kernel_size=(1, 21), padding=(0, 10), groups=dim)
        self.conv_att_2 = nn.Conv2d(dim, dim, kernel_size=(21, 1), padding=(10, 0), groups=dim)
        self.conv_3 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        
        attn = self.conv0(x)
        attn = self.conv_3(self.conv_att_2(self.conv_att_1(attn))+attn)

        x_0 = self.relu(self.conv0_2(self.conv0_1(x)))
        x_1 = self.relu(self.conv1_2(self.conv1_1(x)))
        x_2 = self.relu(self.conv2_2(self.conv2_1(x)))
        x_lkc = self.lkc_fusion(torch.cat([x_0,x_1,x_2], 1))

        return attn*x_lkc
        # return x_lkc
    

class AttentionModule_s(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.relu = nn.ReLU()
        self.skc_conv_1 = nn.Conv2d(dim, dim, 3,1,1)
        self.skc_conv_2 = nn.Conv2d(dim, dim, 3,1,1)

        self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.conv_att_1 = nn.Conv2d(dim, dim, kernel_size=(1, 11), padding=(0, 5), groups=dim)
        self.conv_att_2 = nn.Conv2d(dim, dim, kernel_size=(11, 1), padding=(5, 0), groups=dim)
        self.conv_3 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        
        attn = self.conv_3(self.conv_att_2(self.conv_att_1(x)))

        x_skc = self.skc_conv_2(self.relu(self.skc_conv_1(x)))

        return attn*x_skc
        # return x_skc



# -------------------------
# Spatial Attention Block
# -------------------------
class SpatialAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.ReLU()
        self.spatial_gating_unit = AttentionModule(dim)
        self.proj_2 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        x = self.act(self.proj_1(x))
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x 
    
class SpatialAttention_s(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.ReLU()
        self.spatial_gating_unit = AttentionModule_s(dim)
        self.proj_2 = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        # x = self.act(self.proj_1(x))
        x = self.spatial_gating_unit(x)
        # x = self.proj_2(x)
        return x 



class LKCABlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = SpatialAttention(dim)
        self.norm2 = nn.GroupNorm(1, dim)
        self.ffn = FFN(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
    
class SKCABlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = SpatialAttention_s(dim)
        self.norm2 = nn.GroupNorm(1, dim)
        self.ffn = FFN(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


if __name__ == "__main__":
    x = torch.randn(1, 64, 128, 128).cuda()  # (B, C, H, W)
    block = LKCABlock(dim=64).cuda()
    y = block(x)
    print(y.shape)  # 输出: torch.Size([2, 64, 128, 128])
