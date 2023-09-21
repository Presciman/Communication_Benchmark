#!/bin/bash

#SBATCH --job-name=comm
#SBATCH --nodes=8
#SBATCH -p gpu
#SBATCH --gpus-per-node 4
#SBATCH -A r00114
#SBATCH --time=00:30:00
#SBATCH --output=/N/u/sunbaix/BigRed200/comm_bench/logs/newb8node_%j.out
#SBATCH --error=/N/u/sunbaix/BigRed200/comm_bench/logs/newb8node_%j.err

# Figure out training environment
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export MASTER_PORT=5432
export RANK=$SLURM_PROCID
if [[ -z "${SLURM_NODELIST}" ]]; then
    RANKS=$HOSTNAME
    NNODES=1
else
    NODEFILE=/tmp/nodefile
    scontrol show hostnames $SLURM_NODELIST > $NODEFILE
    MASTER_RANK=$(head -n 1 $NODEFILE)
    RANKS=$(tr '\n' ' ' < $NODEFILE)
    NNODES=$(< $NODEFILE wc -l)
fi

#PRELOAD+="source ~/.bashrc ; "
PRELOAD="module load miniconda/4.12.0 ; "
PRELOAD+="conda activate bert-pytorch ; "
PRELOAD+="cd /N/u/sunbaix/BigRed200/comm_bench ; "
# torchrun launch configuration
LAUNCHER="python3 -m torch.distributed.launch "
#LAUNCHER+="--nnodes=$NNODES --node_rank=$RANK --nproc_per_node=4 --master_addr $MASTER_ADDR --master_port $MASTER_PORT "
# if [[ "$NNODES" -eq 1 ]]; then
#     LAUNCHER+="--standalone "
# else
#     LAUNCHER+="--rdzv_backend=c10d --rdzv_endpoint=$MASTER_RANK "
# fi

CMD="/N/u/sunbaix/BigRed200/comm_bench/run.py "
FULL_CMD=" $PRELOAD $LAUNCHER $CMD $@ "
echo "Training Command: $FULL_CMD"

# Launch the pytorch processes on each worker (use ssh for remote nodes)
RANK=0
for NODE in $RANKS; do
    LAUNCHER+="--nnodes=$NNODES --node_rank=$RANK --nproc_per_node=4 --master_addr $MASTER_ADDR --master_port $MASTER_PORT "
    CMD="run.py "
    FULL_CMD=" $PRELOAD $LAUNCHER $CMD $@ "
    if [[ "$NODE" == "$HOSTNAME" ]]; then
        echo "Launching rank $RANK on local node $NODE"
        eval $FULL_CMD &
    else
        echo "Launching rank $RANK on remote node $NODE"
        ssh $NODE "cd $PWD; $FULL_CMD" &
    fi
    RANK=$((RANK+1))
done

wait
