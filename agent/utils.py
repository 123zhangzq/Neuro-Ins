# -*- coding: utf-8 -*-

import time
import torch
import os
from tqdm import tqdm
from utils.logger import log_to_screen, log_to_tb_val, log_to_screen_and_file
import torch.distributed as dist
from torch.utils.data import DataLoader
from tensorboard_logger import Logger as TbLogger
import random

def gather_tensor_and_concat(tensor):
    gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_t, tensor)
    return torch.cat(gather_t)

def validate(rank, problem, agent, val_dataset, tb_logger, distributed = False, _id = None):
            
    # Validate mode
    if rank==0: print('\nValidating...', flush=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    opts = agent.opts
    if opts.eval_only:
        torch.manual_seed(opts.seed)
        random.seed(opts.seed)
    agent.eval()
    
    val_dataset = problem.make_dataset(filename=opts.val_dataset, size=opts.graph_size,
                               num_samples=opts.val_size,
                               flag_val=True)

    if distributed and opts.distributed:
        device = torch.device("cuda", rank)
        torch.distributed.init_process_group(backend='nccl', world_size=opts.world_size, rank = rank)
        torch.cuda.set_device(rank)
        agent.actor.to(device)
        if torch.cuda.device_count() > 1:
            agent.actor = torch.nn.parallel.DistributedDataParallel(agent.actor,
                                                                   device_ids=[rank])
        if not opts.no_tb and rank == 0:
            tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, 
                                                          opts.graph_size), opts.run_name))

    
    if distributed and opts.distributed:
        assert opts.val_batch_size % opts.world_size == 0
        train_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size = opts.val_batch_size // opts.world_size, shuffle=False,
                                    num_workers=0,
                                    pin_memory=True,
                                    sampler=train_sampler)
    else:
        val_dataloader = DataLoader(val_dataset, batch_size=opts.val_batch_size, shuffle=False,
                                   num_workers=0,
                                   pin_memory=True)
    
    s_time = time.time()

    for batch in tqdm(val_dataloader, desc = 'inference', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
        padded_solution, final_obj, bool_obj_ci, count_obj_ci, average_diff_obj_ci, bool_obj_mm, count_obj_mm, average_diff_obj_mm = agent.rollout(problem,batch, do_sample = False, show_bar = rank==0)


        
    if distributed and opts.distributed: dist.barrier()
    
    if distributed and opts.distributed:
        # initial_cost = gather_tensor_and_concat(cost_hist[:,0].contiguous())
        # time_used = gather_tensor_and_concat(torch.tensor([time.time() - s_time]).cuda())
        # bv = gather_tensor_and_concat(bv.contiguous())
        # costs_history = gather_tensor_and_concat(cost_hist.contiguous())
        # search_history = gather_tensor_and_concat(best_hist.contiguous())
        # reward = gather_tensor_and_concat(r.contiguous())
        pass
    
    else:

        time_used = torch.tensor([time.time() - s_time]) # bs

        
    if distributed and opts.distributed: dist.barrier()
        
    # log to screen  
    if rank == 0: log_to_screen_and_file(time_used,
                                  count_obj_ci, average_diff_obj_ci, count_obj_mm, average_diff_obj_mm,
                                  batch_size = opts.val_size,
                                  dataset_size = len(val_dataset),output_file_path='./print.txt', epoch = _id)
    
    # log to tb
    # if(not opts.no_tb) and rank == 0:
    #     log_to_tb_val(tb_logger,
    #                   time_used,
    #                   initial_cost,
    #                   bv,
    #                   reward,
    #                   costs_history,
    #                   search_history,
    #                   batch_size = opts.val_size,
    #                   val_size =  opts.val_size,
    #                   dataset_size = len(val_dataset),
    #                   T = opts.T_max,
    #                   epoch = _id)
    
    if distributed and opts.distributed: dist.barrier()
    
