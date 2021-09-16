import torch 

def get_num_gpus():
    return torch.cuda.device_count()