import sys
sys.path.append('.')

from src.module.SEVTAttention import SEVTAttention
import logging

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

model_name = 'MSFCN'
logger = logging.getLogger(f'model.{model_name}')

__all__ = [model_name]

def memory_usage(tensor_size):
    max_memory = torch.cuda.max_memory_allocated()
    if max_memory == 0:
        usage = 0
    else:
        usage = torch.cuda.memory_allocated() / max_memory
    logger.debug(f'[memory usage] {usage} {tensor_size}')

 
class FCN(nn.Module):
    def __init__(self, in_channels, scale):
        super().__init__()
        chan_list = [in_channels,64,128,256,128]
        chan_list = [in_channels,64,128,64]
        #ks = [12,8,5,3]
        #stride = [6,4,2,1]
        ks = [7,5,3]
        stride = [4,3,1]
        if scale > 1:
            self.scalepool = nn.AvgPool2d(scale,scale,padding=scale//2)
        else:
            self.scalepool = None
        
        n_layer = len(chan_list)-1
        fcn_layers = []
        for i in range(n_layer):
            fcn_layers += [nn.Conv2d(in_channels=chan_list[i],out_channels=chan_list[i+1],kernel_size=ks[i],stride=stride[i],padding=stride[i],bias=True)]
            fcn_layers += [nn.ReLU()]
        
        self.model = nn.Sequential(*fcn_layers)
        logger.debug(self.model)

        self.out_dim = chan_list[n_layer]
        logger.debug(self.out_dim)

    def forward(self, x):
        logger.debug(f'x shape in fcn {x.shape}')
        if self.scalepool:
            x = self.scalepool(x)
        x = self.model(x)
        logger.debug(f'model x shape in fcn {x.shape}')
        return x

def conv1x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation,
                groups=groups, bias=False, dilation=dilation)

def get_msfcnattn(attn_layer, attn_param, nclass=2, inplane=1, globalpool='maxpool', num_layers=1):
    if attn_layer == 'SEVT':
        attn_layer = SEVTAttention
    else:
        pass
    
    return MSFCN_ATTN(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='twoheadfcnca', attn_layer=attn_layer,attn_param=attn_param)

def get_msfcn_sevt_para(nclass=2, inplane=1, globalpool='maxpool', num_layers=1):
    return MSFCN_ATTN(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='twoheadfcnca', attn_layer=SEVTAttention,attn_param={'structure':'P',})

def get_msfcn_sevt_para1(nclass=2, inplane=1, globalpool='maxpool', num_layers=1):
    return MSFCN_ATTN(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='twoheadfcnca', attn_layer=SEVTAttention,attn_param={'structure':'P1',})

def get_msfcn_sevt_para2(nclass=2, inplane=1, globalpool='maxpool', num_layers=1):
    return MSFCN_ATTN(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='twoheadfcnca', attn_layer=SEVTAttention, attn_param={'structure':'P2',})

def get_msfcn_sevt_serial1(nclass=2, inplane=1, globalpool='maxpool', num_layers=1):
    return MSFCN_ATTN(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='twoheadfcnca', attn_layer=SEVTAttention, attn_param={'structure':'S1',})

def get_msfcn_sevt_serial2(nclass=2, inplane=1, globalpool='maxpool', num_layers=1):
    return MSFCN_ATTN(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='twoheadfcnca', attn_layer=SEVTAttention, attn_param={'structure':'S2',})

def get_msfcn_sharedfcnsevt(nclass=2, inplane=1, globalpool='maxpool', num_layers=1):
    return MSFCN_ATTN(nclass, nc=inplane, globalpool=globalpool, num_layers=num_layers,
            multiscale='sharedfcnca', attn_layer=SEVTAttention, attn_param={'structure':'S2',})

class MSFCN_ATTN(nn.Module):
    def __init__(self, 
        nclass, 
        nc, 
        globalpool, 
        num_layers,
        multiscale,
        attn_layer,
        attn_param=None,
    ):
        super().__init__()
        self.nclass = nclass
        self.nc = nc
        self.globalpool = globalpool
        self.num_layers = num_layers
        self.multiscale = multiscale
        self.attn_param = attn_param
        self.attn_layer = attn_layer

        if self.multiscale == 'twoheadfcn' or self.multiscale == 'twoheadfcnca':
            self.cnn_repre0 = FCN(nc,1)
            self.cnn_repre1 = FCN(nc,2)
        else:
            self.cnn_repre0 = FCN(nc, 1)

        model_dim = self.cnn_repre0.out_dim
        self.dowm = nn.Sequential(
                #conv1x1(self.cnn_repre0.out_dim, down_dim),
                nn.Conv2d(self.cnn_repre0.out_dim, model_dim, kernel_size=3, stride=3, padding=1),
                nn.ReLU()
            )

        if self.multiscale == 'sharedfcnca':
            self.down_sample_input = nn.Sequential(
                nn.AvgPool2d(2,2,padding=2//2)
            )

        self.attn0 = self.attn_layer(model_dim, *attn_param)
        # input 
        if self.multiscale == 'twoheadfcnca':
            self.attn1 = attn_layer(model_dim, *attn_param)
            
        if globalpool == 'maxpool':
            self.global_pool = nn.AdaptiveMaxPool2d(1)
        elif globalpool == 'avgpool':
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        attn_dim = self.attn0.out_dim
        self.flatten = nn.Flatten(1)
        self.classifier = self.make_classifier(attn_dim + attn_dim,nclass)

    def make_classifier(self, hidden_size, nclass, layer_num=1):
        layers = []
        sz = hidden_size
        
        for l in range(layer_num-1):
            layers += [nn.Linear(sz, sz//2)]
            layers += [nn.ReLU(True)]
            layers += [nn.Dropout()]
            sz //= 2
        
        layers += [nn.Dropout(0.5)]
        layers += [nn.Linear(sz, nclass)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        logger.debug(f'x {x.size()}')

        c_in_0 = rearrange(x, f'b t v -> b {self.nc} t v')
        logger.debug(f'c_in_0 size {c_in_0.size()}')

        #memory_usage(c_in.size())
        
        c_out0 = self.cnn_repre0(c_in_0)

        if self.multiscale == 'twoheadfcn' or self.multiscale == 'twoheadfcnca':
            c_out1 = self.cnn_repre1(c_in_0)
        elif self.multiscale == 'sharedfcnca':
            c_in_1 = self.down_sample_input(c_in_0)
            c_out1 = self.cnn_repre0(c_in_1)
        logger.debug(f'c_out.size {c_out0.size()}')
        
        attn_out0 = self.attn0(c_out0)
        
        if self.multiscale == 'twoheadfcn' or self.multiscale == 'sharedfcnca':
            attn_out1 = self.attn0(c_out1)
        elif self.multiscale == 'twoheadfcnca':
            attn_out1 = self.attn1(c_out1)
        
        logger.debug(f'attn_out0 {attn_out0.shape}')
        logger.debug(f'attn_out1 {attn_out1.shape}')

        out0 = self.global_pool(attn_out0)
        out1 = self.global_pool(attn_out1)
        out0 = self.flatten(out0)
        out1 = self.flatten(out1)

        logger.debug(f'globalpool {out0.shape}')
        logger.debug(f'globalpool {out1.shape}')

        out = torch.cat((out0,out1),1)
        logger.debug(f'out {out.shape}')
        
        output = self.classifier(out)
        
        return output

if __name__ == '__main__':
    
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)
    i = torch.ones((1,50,128))
    from torch.profiler import profile, ProfilerActivity, record_function

    patch = 64
    #pcnn = get_msfcn_twoheadfcn(2,1,'maxpool',num_layers=1)
    pcnn = get_msfcn_sharedfcnsevt(2,1,'maxpool',num_layers=1)
    #pcnn = get_msfcn_sharedfcnsevt(2,1,'maxpool',num_layers=1)

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True,with_flops=True) as prof:
        with record_function("model_inference"):
            print(pcnn(i).shape)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))