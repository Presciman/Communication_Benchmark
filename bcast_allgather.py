import os
import torch
import torch.distributed as dist
import time

##initialize
if 'LOCAL_RANK' in os.environ:
    local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
torch.distributed.init_process_group(backend='nccl', init_method='env://')

rank = dist.get_rank()
size = dist.get_world_size()

tensor0=torch.rand([16,16,3,3],dtype=torch.float).to(device)
tensor1=torch.rand([32,32,3,3],dtype=torch.float).to(device)


