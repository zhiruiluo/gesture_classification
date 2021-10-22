import sys
import torch
import torch.nn as nn
from einops import rearrange
import logging

module_name = 'SEVTAttention'
logger = logging.getLogger(f'module.{module_name}')

def conv1x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation,
                groups=groups, bias=False, dilation=dilation)

def INF(B,H,W, device):
     return -torch.diag(torch.tensor(float("inf")).to(device).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, mul_input = True):
        super(SELayer, self).__init__()
        self.mul_input = mul_input
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        logger.debug(f'SElayer {y.shape} y {y.expand_as(x).shape}')
        if self.mul_input:
            return x * y.expand_as(x)
        else:
            return y.expand_as(x)

class vAttn(nn.Module):
    def __init__(self, in_channels, add_input=True):
        super().__init__()
        out_channels = in_channels//8
        self.add_input = add_input
        self.query_w = conv1x1(in_channels,out_channels)
        self.key_w = conv1x1(in_channels,out_channels)
        self.value_w = conv1x1(in_channels,in_channels//2)
        self.attn_value_w = conv1x1(in_channels//2,in_channels)

        self.softmax = nn.Softmax(-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B = x.size(0)
        # x: (B,C,V,T)
        query_v = self.query_w(x) # (B,C,V,T)
        query_v = rearrange(query_v, 'b c v t -> (b t) v c')

        key_v = self.key_w(x)     # (B,C,V,T)
        key_v = rearrange(key_v, 'b c v t -> (b t) c v')
        
        value_v = self.value_w(x) # (B,C1,V,T)
        value_v = rearrange(value_v, 'b c1 v t -> (b t) v c1')

        # softmax at (B T V V)
        attn = self.softmax(torch.bmm(query_v, key_v))
        
        attn_value = rearrange(torch.bmm(attn, value_v), '(b t) v c1 -> b c1 v t', b=B)
        attn_value = self.attn_value_w(attn_value)  # (B,C1,V,T) -> (B,C,V,T)

        if self.add_input:
            return x + self.sigma * attn_value
        else:
            return self.sigma * attn_value


class tAttn(nn.Module):
    def __init__(self, in_channels, add_input=True):
        super().__init__()
        out_channels = in_channels//8
        self.add_input = add_input
        self.query_w =conv1x1(in_channels,out_channels)
        self.key_w = conv1x1(in_channels,out_channels)
        self.value_w = conv1x1(in_channels,in_channels//2)
        self.attn_value_w = conv1x1(in_channels//2,in_channels)

        self.softmax = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B = x.size(0)
        # x: (B,C,V,T)
        query_v = self.query_w(x)  # (B,C,V,T)
        query_v = rearrange(query_v, 'b c v t -> (b v) t c')

        key_v = self.key_w(x)  # (B,C,V,T)
        key_v = rearrange(key_v, 'b c v t -> (b v) c t')

        value_v = self.value_w(x) # (B,C1,V,T)
        value_v = rearrange(value_v, 'b c1 v t -> (b v) t c1')

        # attention softmax at (B V T T)
        attn = self.softmax(torch.bmm(query_v, key_v))

        attn_value = rearrange(torch.bmm(attn, value_v), '(b v) t c1 -> b c1 v t',b=B)
        attn_value = self.attn_value_w(attn_value) # (B C1 V T) (B,C,V,T)

        if self.add_input:
            return x + self.sigma * attn_value #, attn.view(B,V,T,T)
        else:
            return self.sigma * attn_value


class SEVTAttention(nn.Module):
    def __init__(self, in_dim, structure, relu=False):
        super().__init__()
        self.out_dim = in_dim
        self.structure = structure
        if self.structure in ['P1','P2','P3']:
            add_input = False
            mul_input = False
        else:
            add_input = True
            mul_input = True
        self.vattn = vAttn(in_dim,add_input=add_input)
        self.tattn = tAttn(in_dim, add_input=add_input)
        self.selayer = SELayer(in_dim,mul_input=mul_input)

        self.relu = nn.ReLU(inplace=True) if relu else None
        
    def forward(self, x):
        if self.structure == 'P':
            x1 = self.vattn(x)
            x2 = self.tattn(x)
            x3 = self.selayer(x)
            x = x1 + x2 + x3
        elif self.structure == 'P1':
            x1 = self.vattn(x)
            x2 = self.tattn(x)
            x3 = self.selayer(x)
            x = (x1 + x2) + x * x3
        elif self.structure == 'P2':
            x1 = self.vattn(x)
            x2 = self.tattn(x)
            x3 = self.selayer(x)
            x = (x + x1 + x2) * x3
        if self.structure == 'P3':
            x1 = self.vattn(x)
            x2 = self.tattn(x)
            x3 = self.selayer(x)
            x = x1 + x2 + x3 + x
        elif self.structure == 'S1':
            x = self.selayer(x)
            x = self.vattn(x)
            x = self.tattn(x)
        elif self.structure == 'S2':
            x = self.vattn(x)
            x = self.tattn(x)
            x = self.selayer(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

if __name__ == '__main__':

    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)

    input_tensor = torch.randn((1,128,86,44))
    model = SEVTAttention(128,'parallel')
    
    from torch.profiler import profile, ProfilerActivity, record_function

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,with_flops=True) as prof:
        with record_function("model_inference"):
            logger.debug(model(input_tensor).shape)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))