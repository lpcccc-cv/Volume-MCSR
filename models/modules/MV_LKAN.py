import functools
import torch
import torch.nn as nn
import models.modules.module_util as mutil
import einops
from einops import rearrange
import numbers
import torch.nn.functional as F
from models.modules.lkcan import LKCABlock, SKCABlock
try:
    from models.modules.DCNv2.dcn_v2 import DCN_sep
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class LKAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(LKAttention, self).__init__()

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        self.conv0_1 = nn.Conv2d(dim*2, dim, kernel_size=(5, 5), padding=(2, 2), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, kernel_size=(1, 15), padding=(0, 7), groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, kernel_size=(15, 1), padding=(7, 0), groups=dim)

        self.conv3_1 = nn.Conv2d(dim, dim, kernel_size=(1, 1), padding=(0, 0), groups=dim)
        

    def forward(self, x, y):

        assert x.shape == y.shape, 'The shape of feature maps from image and event branch are not equal!'

        q = self.q(x) # image
        k = self.k(y) # ref
        v = self.v(y) # ref

        x_0 = self.conv0_1(torch.cat([q, k], dim=1))
        x_1 = self.conv1_1(x_0)
        x_2 = self.conv2_1(x_1)

        x_attn = self.conv3_1(x_0 + x_2)

        return x_attn*v

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(AttentionTransformerBlock, self).__init__()

        self.norm1_image_1 = LayerNorm(dim, LayerNorm_type)
        self.norm1_image_2 = LayerNorm(dim, LayerNorm_type)
        self.attn = LKAttention(dim, num_heads, bias)
        # ffn
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

    def forward(self, image, ref):
        # image: b, c, h, w
        # event: b, c, h, w
        # return: b, c, h, w

        assert image.shape == ref.shape, 'the shape of image doesnt equal to event'
        b, c , h, w = image.shape
        fused = image + self.attn(self.norm1_image_1(image), self.norm1_image_2(ref)) # b, c, h, w

        # ffn
        fused = to_3d(fused) # b, h*w, c
        fused = fused + self.ffn(self.norm2(fused))
        fused = to_4d(fused, h, w)

        return fused


class DCN_Align(nn.Module):
    def __init__(self, nf=32, groups=4):
        super(DCN_Align, self).__init__()

        self.offset_conv1_1 = nn.Conv2d(nf * 4, nf, 3, 1, 1, bias=True) 
        self.offset_conv2_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # down1    
        self.offset_conv3_1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.offset_conv4_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # down2
        self.offset_conv6_1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.offset_conv7_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True) 
        # up2
        self.offset_conv1_2 = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.offset_conv2_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # up1
        self.offset_conv3_2 = nn.Conv2d(nf*2, nf*2, 3, 1, 1, bias=True)
        # self.offset_conv4_2 = nn.Conv2d(nf, 32, 3, 1, 1, bias=True)

        self.dcnpack = DCN_sep(nf*2, nf*2, 3, stride=1, padding=1, dilation=1,
                            deformable_groups=4)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, fea1, fea2):
        '''align other neighboring frames to the reference frame in the feature level
        estimate offset bidirectionally
        '''
        offset = torch.cat([fea1, fea2], dim=1)
        offset = self.lrelu(self.offset_conv1_1(offset)) 
        offset1 = self.lrelu(self.offset_conv2_1(offset)) 
        # down1
        offset2 = self.lrelu(self.offset_conv3_1(offset1))
        offset2 = self.lrelu(self.offset_conv4_1(offset2))
        # down2   
        offset3 = self.lrelu(self.offset_conv6_1(offset2))
        offset3 = self.lrelu(self.offset_conv7_1(offset3))
        # up1
        offset = F.interpolate(offset3, scale_factor=2, mode='bilinear', align_corners=False)
        offset = self.lrelu(self.offset_conv1_2(torch.cat((offset, offset2), 1))) 
        offset = self.lrelu(self.offset_conv2_2(offset)) 
        # up2
        offset = F.interpolate(offset, scale_factor=2, mode='bilinear', align_corners=False)
        offset = self.offset_conv3_2(torch.cat((offset, offset1), 1))
        # base_offset = self.offset_conv4_2(offset)
 
        aligned_fea = self.dcnpack(fea2, offset)

        return aligned_fea


class ConvGRUCell(nn.Module):
    """单步 ConvGRU 单元"""
    def __init__(self, input_dim, hidden_dim, kernel_size=3, padding=1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.attn = AttentionTransformerBlock(self.hidden_dim, num_heads=1, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias')
        self.DCN_align = DCN_Align(nf=32, groups=4)
        
        self.conv_c = nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size, padding=padding)
        self.conv_h = nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size, padding=padding)

    def forward(self, x, h_prev):
        if h_prev is None:
            h_prev = torch.zeros_like(x[:, :self.hidden_dim, :, :])

        h_prev = self.attn(x, self.DCN_align(x, h_prev))
        # h_prev = self.DCN_align(x, h_prev)
        h = self.conv_h(torch.cat([h_prev,x],1))
        c = self.conv_c(torch.cat([h_prev,x],1))

        return h, c


class ConvGRU(nn.Module):
    """沿时间维的 ConvGRU"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.cell = ConvGRUCell(input_dim, hidden_dim)

    def forward(self, x, reverse=False):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        outputs = []
        h = x[:,0]

        for t in range(1,T-1):
            h,c = self.cell(x[:, t], h)
            outputs.append(c)

        if reverse:
            outputs = outputs[::-1]  # 恢复时间顺序
        return torch.stack(outputs, dim=1)  # (B, T, hidden_dim, H, W)


class BidirectionalRDAN(nn.Module):
    """双向 ConvGRU 模块"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.forward_gru = ConvGRU(input_dim, hidden_dim)
        self.backward_gru = ConvGRU(input_dim, hidden_dim)
        self.fusion = nn.Conv2d(2 * hidden_dim, input_dim, kernel_size=1)

    def forward(self, x):
        # 前向
        out_fwd = self.forward_gru(x)        # (B, T, hidden_dim, H, W)
        # 后向
        out_bwd = self.backward_gru(torch.flip(x, dims=[1]), reverse=True)  # (B, T, hidden_dim, H, W)
        # 融合前后向
        out = torch.cat([out_fwd, out_bwd], dim=2)  # (B, T, 2*hidden_dim, H, W)
        B, T, C2, H, W = out.shape
        out = self.fusion(out.view(B*T, C2, H, W)).view(B, T, -1, H, W)
        # print(out_fwd.shape, out_bwd.shape, out.shape)
        return out


class CMS(nn.Module):
    def __init__(self, nf=64):
        super(CMS, self).__init__()

        self.conv1_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # down1
        self.conv3_1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv4_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # down2
        self.conv6_1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv7_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # --------------------
        # Transposed Convolution Upsampling
        # --------------------
        self.up1 = nn.ConvTranspose2d(
            nf, nf,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=True
        )

        self.up2 = nn.ConvTranspose2d(
            nf, nf,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=True
        )

        # fusion after up1
        self.conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.conv2_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # fusion after up2
        self.conv3_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.conv4_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(
            negative_slope=0.1,
            inplace=True
        )

    def forward(self, fea):

        fea = self.lrelu(self.conv1_1(fea))
        fea1 = self.lrelu(self.conv2_1(fea))

        # down1
        fea2 = self.lrelu(self.conv3_1(fea1))
        fea2 = self.lrelu(self.conv4_1(fea2))

        # down2
        fea3 = self.lrelu(self.conv6_1(fea2))
        fea3 = self.lrelu(self.conv7_1(fea3))

        # up1
        cur_fea = self.lrelu(self.up1(fea3))
        cur_fea = self.lrelu(
            self.conv1_2(torch.cat([cur_fea, fea2], dim=1))
        )
        cur_fea = self.lrelu(self.conv2_2(cur_fea))

        # up2
        cur_fea = self.lrelu(self.up2(cur_fea))
        cur_fea = self.lrelu(
            self.conv3_2(torch.cat([cur_fea, fea1], dim=1))
        )
        cur_fea = self.conv4_2(cur_fea)

        return cur_fea



class MultiViewBlock(nn.Module):
    def __init__(self,n_feat):
        super().__init__()

        self.stage = 4

        self.multi_contrast_fusion_sag = nn.Conv2d(n_feat*2, n_feat, kernel_size=3, padding=1)
        self.multi_contrast_fusion_cor = nn.Conv2d(n_feat*2, n_feat, kernel_size=3, padding=1)

        self.conv_sag = nn.Sequential(*[LKCABlock(dim=64) for i in range(2)])
        self.conv_sag_ref = nn.Sequential(*[SKCABlock(dim=64) for i in range(1)])
        
        self.conv_cor = nn.Sequential(*[LKCABlock(dim=64) for i in range(2)])
        self.conv_cor_ref = nn.Sequential(*[SKCABlock(dim=64) for i in range(1)])

        self.multi_view_fusion = nn.Conv3d(n_feat*2, n_feat, kernel_size=3, padding=1)


    def forward(self,x,ref):

        B, T, C, H, W = x.shape

        ref_sag_f = einops.rearrange(ref,'b t c h w -> (b h) c t w')
        ref_sag_f = self.conv_sag_ref(ref_sag_f)
        x_sag_f = einops.rearrange(x,'b t c h w -> (b h) c t w')
        x_sag_f = self.multi_contrast_fusion_sag(torch.cat([x_sag_f,ref_sag_f],1))
        x_sag_f = self.conv_sag(x_sag_f)
        x_sag_f = einops.rearrange(x_sag_f,'(b h) c t w -> b t c h w',b=B,t=T)
        
        ref_cor_f = einops.rearrange(ref,'b t c h w -> (b w) c t h')
        ref_cor_f = self.conv_cor_ref(ref_cor_f)        
        x_cor_f = einops.rearrange(x,'b t c h w -> (b w) c t h')       
        x_cor_f = self.multi_contrast_fusion_cor(torch.cat([x_cor_f,ref_cor_f],1))
        x_cor_f = self.conv_cor(x_cor_f)
        x_cor_f = einops.rearrange(x_cor_f,'(b w) c t h -> b t c h w',b=B,w=W)
        
        multi_view_fea = torch.cat([x_cor_f,x_sag_f], 2)
        multi_view_fea = einops.rearrange(multi_view_fea, 'b t c h w -> b c t h w')
        x_out = self.multi_view_fusion(multi_view_fea) # b c t h w
        x_out = einops.rearrange(x_out, 'b c t h w -> b t c h w')

        return x_out 
    

    
    
class MV_LKAN(nn.Module):
    def __init__(self, nf=64, scale=None):
        super(MV_LKAN, self).__init__()
        
        self.scale = scale
        self.nf = nf
        input_channel = 1
    
        ResidualBlock_noBN_f = functools.partial(mutil.ResidualBlock_noBN, nf=nf)
        self.conv_first = nn.Conv2d(input_channel, nf, 3, 1, 1, bias=True)
        self.feature_extraction_RFEN = mutil.make_layer(ResidualBlock_noBN_f, 5)

        self.conv_first_ref = nn.Conv2d(input_channel, nf, 3, 1, 1, bias=True)
        self.CMS = CMS()
        self.rec_ref = nn.Conv2d(64, 1, 3, 1, 1, bias=True)

        self.HRconv1 = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last1 = nn.Conv2d(64, input_channel, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
        self.conv_lstm = BidirectionalRDAN(input_dim=nf,hidden_dim=nf)
        # refine 
        self.MultiView_Net = MultiViewBlock(n_feat=nf)

    def select_generate_slices(self, volume, scale):
        indices_to_remove = [i*scale for i in range(200)]
        indices_to_keep = [idx for idx in range(volume.shape[1]) if idx not in indices_to_remove]
        slice_generated = torch.index_select(volume, dim=1, index=torch.tensor(indices_to_keep).to(volume.device))
        return slice_generated
    

    def forward(self, x, y, t=None):  
        # print('*******', x.shape, y.shape) 
        B, N, C, H, W = x.size()  # N input video frames
        b, n, c, h, w = y.size()
        input_x = x


        ######### stage I ###########
        #### REFN for input feature extraction
        L1_fea = self.conv_first(x.view(-1, C, H, W))
        L1_fea = self.feature_extraction_RFEN(L1_fea)
        #### CMS for reference feature extraction
        ref_fea = self.conv_first_ref(y.view(-1, c, h, w))
        ref_fea = self.CMS(ref_fea)
        rec_ref = self.rec_ref(ref_fea)
        rec_ref = self.select_generate_slices(rec_ref.view(b, n, -1, h, w), scale = self.scale)
        
        L1_fea = L1_fea.view(B, N, -1, H, W)
        ref_fea = ref_fea.view(b, n, -1, h, w)

        ######### stage II ###########
        # Bi-RDAN for feature compensation
        lstm_fea = [L1_fea[:, 0:1]]
        for i in range(N-1):
            lstm_fea.append(self.conv_lstm(torch.cat((L1_fea[:, i:(i+1)],ref_fea[:,(i*self.scale+1):(i+1)*self.scale],L1_fea[:, (i+1):(i+2)]),1)))
            lstm_fea.append(L1_fea[:, (i+1):(i+2)])

        lstm_fea = torch.cat(lstm_fea, 1)  # b,t,c,h,w
        # print(lstm_fea.shape)

        ######### stage III ###########
        # MultiView LKCA refine
        stage2_fea = self.MultiView_Net(lstm_fea,ref_fea)+lstm_fea
        # stage2_fea = lstm_fea
        B, T, C, H, W = stage2_fea.size()
        stage2_fea = stage2_fea.reshape(B*T, C, H, W) 

        # rec
        out1 = self.conv_last1(self.lrelu(self.HRconv1(stage2_fea)))
        _, _, K, G = out1.size()
        outs_volume = out1.view(B, -1, 1, K, G)

        # post-process: 
        outs_volume[:,::self.scale] = input_x
        
        # select geberated slices
        generated_slices = self.select_generate_slices(outs_volume, scale = self.scale)
        # print('post-process:',outs1.shape)
    
        return generated_slices, rec_ref
    
