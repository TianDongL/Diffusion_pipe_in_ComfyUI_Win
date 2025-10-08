import torch.multiprocessing as mp
import os
import torch

from src.trainer.train_lumina import train_lumina
if __name__ == "__main__":
    
    world_size = torch.cuda.device_count()

    # Use spawn method for starting processes
    mp.spawn(
        train_lumina,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )