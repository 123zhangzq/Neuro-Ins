import os
from tqdm import tqdm
import warnings
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboard_logger import Logger as TbLogger
import random

from utils.utils import clip_grad_norms, rotate_tensor
from nets.actor_network import Actor
from nets.critic_network import Critic
from utils.utils import torch_load_cpu, get_inner_model, move_to, move_to_cuda, pad_solution
from utils.logger import log_to_tb_train
from agent.utils import validate

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []  
        self.obj = []

        
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.obj[:]


class Reinforce:
    def __init__(self, problem_name, size, opts):
        # figure out the options
        self.opts = opts
        
        # figure out the actor
        self.actor = Actor(
            problem_name = problem_name,
            embedding_dim = opts.embedding_dim,
            hidden_dim = opts.hidden_dim,
            n_heads_actor = opts.actor_head_num,
            n_layers = opts.n_encode_layers,
            normalization = opts.normalization,
            v_range = opts.v_range,
            seq_length = size + 1
        )
        
        if not opts.eval_only:
        
            # figure out the critic
            self.critic = Critic(
                    problem_name = problem_name,
                    embedding_dim = opts.embedding_dim,
                    hidden_dim = opts.hidden_dim,
                    n_heads = opts.critic_head_num,
                    n_layers = opts.n_encode_layers,
                    normalization = opts.normalization
                )
        
            # figure out the optimizer
            self.optimizer = torch.optim.Adam(
            [{'params':  self.actor.parameters(), 'lr': opts.lr_model}] + 
            [{'params':  self.critic.parameters(), 'lr': opts.lr_critic}])
            
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, opts.lr_decay, last_epoch=-1,)

        print(f'Distributed: {opts.distributed}')
        if opts.use_cuda and not opts.distributed:
            
            self.actor.to(opts.device)
            if not opts.eval_only: self.critic.to(opts.device)
                
    
    def load(self, load_path):
        
        assert load_path is not None
        load_data = torch_load_cpu(load_path)
        # load data for actor
        model_actor = get_inner_model(self.actor)
        model_actor.load_state_dict({**model_actor.state_dict(), **load_data.get('actor', {})})
        
        if not self.opts.eval_only:
            # load data for critic
            model_critic = get_inner_model(self.critic)
            model_critic.load_state_dict({**model_critic.state_dict(), **load_data.get('critic', {})})
            # load data for optimizer
            self.optimizer.load_state_dict(load_data['optimizer'])
            # load data for torch and cuda
            torch.set_rng_state(load_data['rng_state'])
            if self.opts.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # done
        print(' [*] Loading data from {}'.format(load_path))
        
    
    def save(self, epoch):
        print('Saving model and state...')
        torch.save(
            {
                'actor': get_inner_model(self.actor).state_dict(),
                'critic': get_inner_model(self.critic).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(self.opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )
    
    
    def eval(self):
        torch.set_grad_enabled(False)
        self.actor.eval()
        if not self.opts.eval_only: self.critic.eval()
        
    def train(self):
        torch.set_grad_enabled(True)
        self.actor.train()
        if not self.opts.eval_only: self.critic.train()
    
    def rollout(self, problem, batch, do_sample = False, show_bar = False):     # TODO NOW: output
        batch = move_to(batch, self.opts.device) # batch_size, graph_size, 2
        bs, gs, dim = batch['coordinates'].size()


        batch_feature = problem.input_feature_encoding(batch)



        solutions = move_to(problem.get_static_solutions(batch), self.opts.device).long()
        obj = problem.get_costs(batch, solutions, flag_finish=False)
        padded_solution = pad_solution(solutions, batch_feature.size(1))

        reward = []

        dy_size = problem.size - 2 * problem.static_orders
        action_his = torch.zeros_like(padded_solution, dtype=torch.bool, device=padded_solution.device)
        for t in tqdm(range(dy_size // 2), disable = self.opts.no_progress_bar or not show_bar, desc = 'rollout', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            step_info = (dy_size, t)
            # pass through model
            exchange = self.actor(problem,
                                  batch_feature,
                                  padded_solution,
                                  action_his,
                                  step_info,
                                  do_sample = do_sample)[0]

            # new solution
            padded_solution, rewards, obj = problem.step(batch, padded_solution, exchange, obj, None)




        # statistic
        obj1 = problem.get_costs(batch, padded_solution, flag_finish=True)
        final_obj = obj.view(-1)
        cheapest_ins_obj = batch['ci_obj'].view(-1)
        mm_obj = batch['mm_obj'].view(-1)

        bool_obj_ci = final_obj.view(-1, 1) < cheapest_ins_obj.view(-1, 1)
        bool_obj_mm = final_obj.view(-1, 1) < mm_obj.view(-1, 1)
        count_obj_ci = torch.sum(final_obj <= cheapest_ins_obj)
        count_obj_mm = torch.sum(final_obj <= mm_obj)

        sum_diff_obj_ci = torch.sum((final_obj - cheapest_ins_obj)/cheapest_ins_obj)
        average_diff_obj_ci = sum_diff_obj_ci / final_obj.size(0)
        sum_diff_obj_mm = torch.sum((final_obj - mm_obj)/mm_obj)
        average_diff_obj_mm = sum_diff_obj_mm / final_obj.size(0)


        out = (padded_solution, # bs, gs
               final_obj, # batch_size, 1
               bool_obj_ci,  # batch_size, 1
               count_obj_ci, # 1
               average_diff_obj_ci,  # 1
               bool_obj_mm,
               count_obj_mm,
               average_diff_obj_mm
               )
        
        return out
    
    
    def start_inference(self, problem, val_dataset, tb_logger):
        if self.opts.distributed:            
            mp.spawn(validate, nprocs=self.opts.world_size, args=(problem, self, val_dataset, tb_logger, True))
        else:
            validate(0, problem, self, val_dataset, tb_logger, distributed = False)
            
    def start_training(self, problem, train_dataset, val_dataset, tb_logger):
        if self.opts.distributed:
            mp.spawn(train, nprocs=self.opts.world_size, args=(problem, self, train_dataset, val_dataset, tb_logger))
        else:
            train(0, problem, self, train_dataset, val_dataset, tb_logger)

            self.optimizer.state

            
def train(rank, problem, agent, train_dataset, val_dataset, tb_logger):
    
    opts = agent.opts  

    warnings.filterwarnings("ignore")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if opts.resume is None:
        torch.manual_seed(opts.seed)
        random.seed(opts.seed)
        
    if opts.distributed:
        device = torch.device("cuda", rank)
        torch.distributed.init_process_group(backend='nccl', world_size=opts.world_size, rank = rank)
        torch.cuda.set_device(rank)
        agent.actor.to(device)
        agent.critic.to(device)
        for state in agent.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        

        agent.actor = torch.nn.parallel.DistributedDataParallel(agent.actor,
                                                               device_ids=[rank])
        if not opts.eval_only: agent.critic = torch.nn.parallel.DistributedDataParallel(agent.critic,
                                                               device_ids=[rank])
        if not opts.no_tb and rank == 0:
            tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, 
                                                          opts.graph_size), opts.run_name))
    else:
        for state in agent.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(opts.device)
                        
    if opts.distributed: dist.barrier()
    
    # Start the actual training loop
    for epoch in range(opts.epoch_start, opts.epoch_end):
        
        agent.lr_scheduler.step(epoch)
        
        # Training mode
        if rank == 0:
            print('\n\n')
            print("|",format(f" Training epoch {epoch} ","*^60"),"|")
            print("Training with actor lr={:.3e} critic lr={:.3e} for run {}".format(agent.optimizer.param_groups[0]['lr'], 
                                                                                 agent.optimizer.param_groups[1]['lr'], opts.run_name) , flush=True)
        # prepare training data
        training_dataset = problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size,filename=train_dataset)
        if opts.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset, shuffle=False)
            training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size // opts.world_size, shuffle=False,
                                            num_workers=0,
                                            pin_memory=True,
                                            sampler=train_sampler)
        else:
            training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, shuffle=False,
                                                       num_workers=0,
                                                       pin_memory=True)
            
        # start training
        step = epoch * (opts.epoch_size // opts.batch_size)  
        pbar = tqdm(total = (opts.K_epochs) * (opts.epoch_size // opts.batch_size) * (opts.T_train // opts.n_step) ,
                    disable = opts.no_progress_bar or rank!=0, desc = 'training',
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        for batch_id, batch in enumerate(training_dataloader):
            train_batch(rank,
                        problem,
                        agent,
                        epoch,
                        step,
                        batch,
                        tb_logger,
                        opts,
                        pbar,
                        )
            step += 1
        pbar.close()
        
        # save new model after one epoch  
        if rank == 0 and not opts.distributed: 
            if not opts.no_saving and (( opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or \
                        epoch == opts.epoch_end - 1): agent.save(epoch)
        elif opts.distributed and rank == 1:
            if not opts.no_saving and (( opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or \
                        epoch == opts.epoch_end - 1): agent.save(epoch)
            
        
        # validate the new model   
        if rank == 0 and not opts.distributed: validate(rank, problem, agent, val_dataset, tb_logger, _id = epoch)
        if rank == 0 and opts.distributed: validate(rank, problem, agent, val_dataset, tb_logger, _id = epoch)
        
        # syn
        if opts.distributed: dist.barrier()

    
def train_batch(
        rank,
        problem,
        agent,
        epoch,
        step,
        batch,
        tb_logger,
        opts,
        pbar,
        ):

    # setup
    agent.train()
    memory = Memory()

    # prepare the input
    batch = move_to_cuda(batch, rank) if opts.distributed else move_to(batch, opts.device)# batch_size, graph_size, 2
    batch_feature = problem.input_feature_encoding(batch).cuda() if opts.distributed \
                        else move_to(problem.input_feature_encoding(batch), opts.device)
    batch_size = batch_feature.size(0)
    exchange = move_to_cuda(torch.tensor([-1,-1,-1]).repeat(batch_size,1), rank) if opts.distributed \
                        else move_to(torch.tensor([-1,-1,-1]).repeat(batch_size,1), opts.device)
    

    # print(f"rank {rank}, data from {batch['id'][0]},{batch['id'][1]} , to {batch['id'][-2]},{batch['id'][-1]}")

    # initial solution of the static orders

    solution = move_to_cuda(problem.get_static_solutions(batch),rank) if opts.distributed \
                        else move_to(problem.get_static_solutions(batch), opts.device)
    obj = problem.get_costs(batch, solution, flag_finish = False)
    padded_solution = pad_solution(solution, batch_feature.size(1))

    # params for training
    gamma = opts.gamma
    n_step = opts.n_step
    T = opts.T_train
    K_epochs = opts.K_epochs
    eps_clip = opts.eps_clip
    epsilon = opts.epsilon  # e-greedy for decoder sampling action
    epsilon_decay = opts.epsilon_decay
    epsilon_info = (epsilon, epsilon_decay, epoch)

    initial_cost = obj
    
    # sample trajectory

    memory.actions.append(exchange)


    # for first step
    entropy = []
    bl_val_detached = []


    dy_size = problem.size - 2 * problem.static_orders
    t = 0
    log_likelihood = 0
    R = 0
    action_his = torch.zeros_like(padded_solution, dtype=torch.bool, device=padded_solution.device)
    while t < (dy_size // 2):

        memory.states.append(padded_solution)


        # get model output
        step_info = (dy_size, t)
        exchange, log_lh, _to_critic, entro_p, CI_action = agent.actor(problem,
                                                             batch_feature,
                                                             padded_solution,
                                                             action_his,
                                                             step_info,
                                                             epsilon_info=epsilon_info,
                                                             do_sample = True,
                                                             require_entropy = True,# take same action
                                                             to_critic = True)

        memory.actions.append(exchange)
        memory.logprobs.append(log_lh)
        log_likelihood += log_lh

        memory.obj.append(obj.unsqueeze(-1))


        entropy.append(entro_p.detach().cpu())


        # state transient
        padded_solution, rewards, obj = problem.step(batch, padded_solution, exchange, obj, CI_action)
        memory.rewards.append(rewards)
        # memory.mask_true = memory.mask_true + info['swaped']

        # store info
        R += rewards

        # next step
        t = t + 1



    # begin update        =======================



    # calculate loss
    loss = (-R * log_likelihood).mean()

    # update gradient step
    agent.optimizer.zero_grad()
    loss.backward()

    # Clip gradient norm and get (clipped) gradient norms for logging
    current_step = int(1)


    # perform gradient descent
    agent.optimizer.step()

    # Logging to tensorboard
    if(not opts.no_tb) and rank == 0:
        if (current_step + 1) % int(opts.log_step) == 0:
            log_to_tb_train(tb_logger, agent, R, log_likelihood, initial_cost, current_step + 1)

    if rank == 0: pbar.update(1)


    # end update
    memory.clear_memory()

