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
tsr_size=(4096,4096)
'''
#Allreduce
tsr_size=(4096,4096)


#warmup
random_tensor = torch.rand(tsr_size,dtype=torch.float).to(device)
dist.all_reduce(random_tensor,op=dist.ReduceOp.SUM)
random_tensor /= size

#allreduce
times=[]
for j in range(7):
    random_tensor = torch.rand(tsr_size,dtype=torch.float).to(device)
    if rank==0:
        print(4096*4096*random_tensor.element_size())
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    dist.barrier()
    #torch.cuda.synchronize()
    start.record()
    dist.all_reduce(random_tensor,op=dist.ReduceOp.SUM)
    random_tensor /= size
    end.record()
    dist.barrier()
    torch.cuda.synchronize()
    #print("{},{};{};{}".format(rank,size,tsr_size,start.elapsed_time(end)))
    #torch.cuda.synchronize()
    comm_time = start.elapsed_time(end)

    #Collect time

    time_tensor = torch.tensor([comm_time],dtype=torch.double).to(device)
    dist.all_reduce(time_tensor,op=dist.ReduceOp.MAX)
    times.append(time_tensor.item())
    if rank == 0:
        print("Rank,GPUs,DataSize,Time")
        print("{},{};{};{}".format(rank,size,tsr_size,time_tensor.item()))
'''

'''
import math
#allgather
#warmup
tensor_list = [torch.zeros(tsr_size, dtype=torch.float).to(device) for _ in range(size)]
random_tensor = torch.rand(tsr_size,dtype=torch.float).to(device)
init_tensor = torch.zeros(tsr_size, dtype=torch.float).to(device)
dist.all_gather(tensor_list,random_tensor)
for i in range(size):
    init_tensor = torch.add(init_tensor,tensor_list[i])
init_tensor /= size
print("Shape: {}".format(init_tensor.size()))

tensor_list = [torch.zeros(tsr_size, dtype=torch.float).to(device) for _ in range(size)]
random_tensor = torch.rand(tsr_size,dtype=torch.float).to(device)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
dist.barrier()
start.record()
dist.all_gather(tensor_list,random_tensor)
for i in range(math.ceil(math.log(size))):
    init_tensor = torch.add(init_tensor,tensor_list[i])
init_tensor /= size
end.record()
dist.barrier()
torch.cuda.synchronize()
comm_time = start.elapsed_time(end)

#Collect time
time_tensor = torch.tensor([comm_time],dtype=torch.double).to(device)
dist.all_reduce(time_tensor,op=dist.ReduceOp.MAX)
print("Rank,GPUs,DataSize,Time")
print("{},{};{};{}".format(rank,size,tsr_size,time_tensor.item()))
'''

'''
import math
#allgather
#warmup

tensor_list = [torch.zeros(tsr_size, dtype=torch.float).to(device) for _ in range(size)]
random_tensor = torch.rand(tsr_size,dtype=torch.float).to(device)
init_tensor = torch.zeros(tsr_size, dtype=torch.float).to(device)
dist.all_gather(tensor_list,random_tensor)
for i in range(size):
    init_tensor = torch.add(init_tensor,tensor_list[i])
init_tensor /= size
print("Shape: {}".format(init_tensor.size()))

CR=2*size
tsr_size1 = (4096//CR,4096)
times=[]
activities=[]
activities.append(torch.profiler.ProfilerActivity.CPU)
activities.append(torch.profiler.ProfilerActivity.CUDA)
with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=7, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('/grand/sbi-fair/allgather/trace_4node/'),
            record_shapes=True,
            with_stack=True) as prof:
    for j in range(7):
        tensor_list = [torch.zeros(tsr_size, dtype=torch.float).to(device) for _ in range(size)]
        tensor_list1 = [torch.zeros(tsr_size1, dtype=torch.float).to(device) for _ in range(size)]
        random_tensor = torch.rand(tsr_size1,dtype=torch.float).to(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        dist.barrier()
        start.record()
        dist.all_gather(tensor_list1,random_tensor)
        for i in range(math.ceil(math.log(size))):
            init_tensor = torch.add(init_tensor,tensor_list[i])
        init_tensor /= size
        end.record()
        dist.barrier()
        torch.cuda.synchronize()
        comm_time = start.elapsed_time(end)

        #Collect time
        time_tensor = torch.tensor([comm_time],dtype=torch.double).to(device)
        dist.all_reduce(time_tensor,op=dist.ReduceOp.MAX)
        times.append(time_tensor.item())
        prof.step()
        if rank == 0:
            print("Run,Rank,GPUs,DataSize,Time,CR")
            print("{};{};{};{};{};{}".format(j,rank,size,tsr_size,time_tensor.item(),CR))
    
if rank == 0:
    print(times)
'''

import math
#allgather
#warmup

random_tensor = torch.rand(tsr_size,dtype=torch.float).cuda()
#print('1')
dist.broadcast(random_tensor, 0, async_op=False)
del random_tensor
torch.cuda.empty_cache()
#rint('2')
#CR=2*size
if rank == 0:
    print("GPUs,DataSize,Time,Beishu,Bytes")
for CR in [0.00897217,0.00390625,0.015625,0.03515625,0.0625,0.125,0.140625,0.25,0.5,0.5625,1,2,4,8,16,32,64,128,256]:
    tsr_size1 = (int(1024*CR),1024)
    times=[]
    # activities=[]
    # activities.append(torch.profiler.ProfilerActivity.CPU)
    # activities.append(torch.profiler.ProfilerActivity.CUDA)
    '''
    with torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=7, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('/grand/sbi-fair/allgather/trace_4node_bcast/'),
                record_shapes=True,
                with_stack=True) as prof:
            '''
    for j in range(10):
        random_tensor = torch.rand(tsr_size1,dtype=torch.float).cuda()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        dist.barrier()
        start.record()
        dist.broadcast(random_tensor, 0, async_op=False)
        end.record()
        dist.barrier()
        torch.cuda.synchronize()
        comm_time = start.elapsed_time(end)

        #Collect time
        time_tensor = torch.tensor([comm_time],dtype=torch.double).cuda()
        dist.all_reduce(time_tensor,op=dist.ReduceOp.MAX)
        times.append(time_tensor.item())
        del random_tensor
        torch.cuda.empty_cache()
        #prof.step()
        #if rank == 0:
            #print("Run,Rank,GPUs,DataSize,Time,CR")
            #print("{};{};{};{};{};{}".format(j,rank,size,tsr_size,time_tensor.item(),CR))
    if rank == 0:
        random_tensor1 = torch.rand(tsr_size1,dtype=torch.float).cuda()
        print("{};{};{};{};{}".format(size,tsr_size1,sum(times)/len(times),CR,random_tensor1.numel()*random_tensor1.element_size()))
        del random_tensor1
        torch.cuda.empty_cache()