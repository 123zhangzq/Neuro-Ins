from torch.utils.data import Dataset
import torch
import pickle
import os

class PDTSP(object):

    NAME = 'pdtsp'  #Pickup and Delivery TSP
    
    def __init__(self, p_size, sta_orders, init_val_met = 'p2d', with_assert = False):
        
        self.size = p_size          # the number of nodes in PDTSP 
        self.static_orders = sta_orders   # the number of static orders in PDTSP
        self.do_assert = with_assert
        self.init_val_met = init_val_met
        self.state = 'eval'
        print(f'PDTSP with {self.size} nodes.', 
              ' Do assert:', with_assert,)
    
    def input_feature_encoding(self, batch):
        return torch.cat([batch['coordinates'], batch['dynamic_loc']], dim=1)

    
    def get_visited_order_map(self, visited_time, step_info):
        dy_size, dy_t = step_info
        bs, gs = visited_time.size()
        valid_l = gs - dy_size + 2 * dy_t
        visited_time = visited_time % valid_l

        return visited_time.view(bs, gs, 1) > visited_time.view(bs, 1, gs)

        
    def get_real_mask(self, selected_node, visited_order_map, step_info, action_his):
        dy_size, dy_t = step_info
        bs, gs, _ = visited_order_map.size()

        # mask the selected nodes
        selected_node_corresponding = torch.where(selected_node > 2 * self.static_orders, selected_node + dy_size // 2, selected_node + self.static_orders)

        mask = visited_order_map.clone()



        mask[torch.arange(bs), selected_node.view(-1)] = True
        mask[torch.arange(bs), selected_node_corresponding.view(-1)] = True
        mask[torch.arange(bs),:,selected_node.view(-1)] = True
        mask[torch.arange(bs),:,selected_node_corresponding.view(-1)] = True

        # mask the un-inserted dynamic orders
        dy_orders_ind = gs - dy_size
        dy_orders_end = gs - dy_size + dy_size // 2
        # un_insert_dy_orders_co = un_insert_dy_orders + dy_size // 2
        #
        # index_range = torch.arange(un_insert_dy_orders, dy_orders_end)
        # index_range_o = torch.arange(un_insert_dy_orders_co, gs)

        index_un_ins = torch.zeros_like(action_his, dtype=torch.bool, device=action_his.device)
        index_un_ins[:, dy_orders_ind:] = action_his[:, dy_orders_ind:] == False
        index_un_ins_row = index_un_ins.unsqueeze(2).expand(-1, -1, gs)
        index_un_ins_col = index_un_ins.unsqueeze(1).expand(-1, gs, -1)
        mask_index_un_ins = torch.logical_or(index_un_ins_row, index_un_ins_col)

        mask[mask_index_un_ins] = True
        return mask

    def get_static_solutions(self, batch):
        assert batch['sol_static'].shape[1] == 2 * self.static_orders + 1, "The input (static orders' routes) is wrong..."
        return batch['sol_static'].to(torch.int64)

    def get_MM_solutions(self, batch):
        assert batch['sol_MM'].shape[1] == self.size + 1, "The input (solution routes) is wrong..."
        return batch['sol_MM'].to(torch.int64)

    def get_initial_solutions(self, batch, val_m = 1):
        
        batch_size = batch['coordinates'].size(0)
    
        def get_solution(methods):
            
            half_size = self.size// 2
            
            if methods == 'random':
                candidates = torch.ones(batch_size,self.size + 1).bool()
                candidates[:,half_size + 1:] = 0
                rec = torch.zeros(batch_size, self.size + 1).long()
                selected_node = torch.zeros(batch_size, 1).long()
                candidates.scatter_(1, selected_node, 0)
                
                for i in range(self.size):
                    dists = torch.ones(batch_size, self.size + 1)
                    dists.scatter_(1, selected_node, -1e20)
                    dists[~candidates] = -1e20
                    dists = torch.softmax(dists, -1)
                    next_selected_node = dists.multinomial(1).view(-1,1)
                    
                    add_index = (next_selected_node <= half_size).view(-1)
                    pairing = next_selected_node[next_selected_node <= half_size].view(-1,1) + half_size
                    candidates[add_index] = candidates[add_index].scatter_(1, pairing, 1)
                    
                    rec.scatter_(1,selected_node, next_selected_node)
                    candidates.scatter_(1, next_selected_node, 0)
                    selected_node = next_selected_node
                    
                return rec
            
            elif methods == 'greedy':
                
                candidates = torch.ones(batch_size,self.size + 1).bool()
                candidates[:,half_size + 1:] = 0
                rec = torch.zeros(batch_size, self.size + 1).long()
                selected_node = torch.zeros(batch_size, 1).long()
                candidates.scatter_(1, selected_node, 0)
                
                for i in range(self.size):
                    
                    d1 = batch['coordinates'].cpu().gather(1, selected_node.unsqueeze(-1).expand(batch_size, self.size + 1, 2))
                    d2 = batch['coordinates'].cpu()
                    
                    dists = (d1 - d2).norm(p=2, dim=2)
                    # dists = batch['dist'].cpu().gather(1,selected_node.view(batch_size,1,1).expand(batch_size, 1, self.size + 1)).squeeze().clone()
                    dists.scatter_(1, selected_node, 1e6)
                    dists[~candidates] = 1e6
                    next_selected_node = dists.min(-1)[1].view(-1,1)
                    
                    add_index = (next_selected_node <= half_size).view(-1)
                    pairing = next_selected_node[next_selected_node <= half_size].view(-1,1) + half_size
                    candidates[add_index] = candidates[add_index].scatter_(1, pairing, 1)
                    
                    rec.scatter_(1,selected_node, next_selected_node)
                    candidates.scatter_(1, next_selected_node, 0)
                    selected_node = next_selected_node

                cost_ = self.get_costs(batch, rec)

                return rec
                
            else:
                raise NotImplementedError()

        return get_solution(self.init_val_met).expand(batch_size, self.size + 1).clone()

    def step(self, batch, rec, exchange, last_obj, CI_action=None):
        if CI_action == None:
            bs, gs = rec.size()

            selected = exchange[:, 0].view(bs, 1)
            first = exchange[:, 1].view(bs, 1)
            second = exchange[:, 2].view(bs, 1)

            next_state = self.insert_star(rec, selected, first, second)

            new_obj = self.get_costs(batch, next_state)

            reward = - (new_obj - last_obj)

            return next_state, reward, new_obj
        else:
            bs, gs = rec.size()

            selected = exchange[:, 0].view(bs, 1)
            first = exchange[:, 1].view(bs, 1)
            second = exchange[:, 2].view(bs, 1)

            next_state = self.insert_star(rec, selected, first, second)

            new_obj = self.get_costs(batch, next_state)

            # CI
            selected_CI = CI_action[:, 0].view(bs, 1)
            first_CI = CI_action[:, 1].view(bs, 1)
            second_CI = CI_action[:, 2].view(bs, 1)

            next_state_CI = self.insert_star(rec, selected_CI, first_CI, second_CI)
            CI_obj = self.get_costs(batch, next_state_CI)



            reward = (CI_obj - last_obj) - (new_obj - last_obj)

            return next_state, reward, new_obj

    def insert_star(self, solution, pair_index, first, second):
        
        rec = solution.clone()

        dy_size = self.size - 2 * self.static_orders

        # fix connection for pairing node
        post_second = rec.gather(1,second)
        rec.scatter_(1,second, pair_index + dy_size // 2)
        rec.scatter_(1,pair_index + dy_size // 2, post_second)
        
        post_first = rec.gather(1,first)
        rec.scatter_(1,first, pair_index)
        rec.scatter_(1,pair_index, post_first)        
        
        return rec
        
    def check_feasibility(self, rec):
        
        problem_size = self.size
        static_pos = 2 * self.static_orders
        d_size = problem_size - static_pos

        assert (
            (torch.arange(problem_size + 1, out=rec.new())).view(1, -1).expand_as(rec)  ==
            rec.sort(1)[0]
        ).all(), ((
            (torch.arange(problem_size + 1, out=rec.new())).view(1, -1).expand_as(rec)  ==
            rec.sort(1)[0]
        ),"not visiting all nodes", rec)
        
        # calculate visited time
        bs = rec.size(0)
        visited_time = torch.zeros((bs,problem_size),device = rec.device)
        pre = torch.zeros((bs),device = rec.device).long()
        for i in range(problem_size):
            visited_time[torch.arange(bs),rec[torch.arange(bs),pre] - 1] = i + 1
            pre = rec[torch.arange(bs),pre]
        # static orders
        assert (
            visited_time[:, 0: static_pos // 2] <
            visited_time[:, static_pos // 2:static_pos]
        ).all(), (visited_time[:, 0: static_pos // 2] <
            visited_time[:, static_pos // 2:static_pos],"static orders delivery without pick-up")
        # dynamic orders
        assert (
            visited_time[:, static_pos: static_pos + d_size // 2] <
            visited_time[:, static_pos + d_size // 2:]
        ).all(), (visited_time[:, static_pos: static_pos + d_size // 2] <
            visited_time[:, static_pos + d_size // 2:],"dynamic orders delivery without pick-up")
        print("check all the feasibilities!!!")
    
    
    def get_swap_mask(self, selected_node, visited_order_map, step_info, action_his):
        return self.get_real_mask(selected_node, visited_order_map, step_info, action_his)
        
    
    def get_costs(self, batch, rec, flag_finish=False):
        # can only get  1) the static orders cost and 2) the final routes with all dynamic orders cost
        batch_size, size = rec.size()
        all_coor = torch.cat([batch['coordinates'], batch['dynamic_loc']], dim=1)

        # check feasibility
        # if self.do_assert:
        #     if flag_finish == True:
        #         self.check_feasibility(rec)
        if flag_finish == True:
            self.check_feasibility(rec)




        if size == 2 * self.static_orders + 1:
            # calculate obj value
            d1 = batch['coordinates'].gather(1, rec.long().unsqueeze(-1).expand(batch_size, size, 2))
            d2 = batch['coordinates'].clone()
            # not return to depot, so remove the distances from last customer to depot. For padding 0, not calculate the distance
            zero_indices = torch.nonzero(rec == 0)[:, 1].view(-1, 1)
            d2[torch.arange(d1.size(0)).view(-1, 1), zero_indices.view(-1, 1), :] = \
                d1[torch.arange(d1.size(0)).view(-1, 1), zero_indices.view(-1, 1), :]

            length = (d1 - d2).norm(p=2, dim=2).sum(1)

            return length
        else:
            # calculate obj value
            d1 = all_coor.gather(1, rec.long().unsqueeze(-1).expand(batch_size, size, 2))
            d2 = all_coor.clone()
            # not return to depot, so remove the distances from last customer to depot. For padding 0, not calculate the distance
            zero_indices = torch.nonzero(rec == 0)
            row_indices = zero_indices[:, 0]
            col_indices = zero_indices[:, 1]
            d2[row_indices, col_indices, :] = d1[row_indices, col_indices, :]

            length =  (d1  - d2).norm(p=2, dim=2).sum(1)

            return length

    @staticmethod
    def make_dataset(*args, **kwargs):
        return PDPDataset(*args, **kwargs)


class PDPDataset(Dataset):
    def __init__(self, filename=None, size=20, num_samples=10000, offset=0, distribution=None, flag_val=False):
        
        super(PDPDataset, self).__init__()
        
        self.data = []
        self.size = size

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl', 'file name error'
            
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            if flag_val:
                self.data = [self.make_val_instance(args) for args in data[offset:offset+num_samples]]
            else:
                self.data = [self.make_instance(args) for args in data[offset:offset+num_samples]]

        else:
            assert filename is not None, 'filename should not be None, please give the path for training dataset'

            # self.data = [{
            #         'loc': torch.FloatTensor(self.size, 2).uniform_(0, 1),
            #         'depot': torch.FloatTensor(2).uniform_(0, 1)} for i in range(num_samples)]
        
        self.N = len(self.data)
        
        # prepare the training instances
        for i, instance in enumerate(self.data):
            self.data[i]['coordinates'] = torch.cat((instance['depot'].reshape(1, 2), instance['loc']),dim=0)
            del self.data[i]['depot']
            del self.data[i]['loc']
        print(f'{self.N} instances initialized.')
    
    def make_instance(self, args):
        depot, loc, sol_static, dynamic_loc, *args = args
        depot = [x / 100 for x in depot]
        loc = [[x / 100 for x in inner_list] for inner_list in loc]
        dynamic_loc = [[x / 100 for x in inner_list] for inner_list in dynamic_loc]


        if len(args) > 0:
            temp = args
        return {
            'loc': torch.tensor(loc, dtype=torch.float),
            'depot': torch.tensor(depot, dtype=torch.float),
            'sol_static': torch.tensor(sol_static, dtype=torch.int),
            'dynamic_loc': torch.tensor(dynamic_loc, dtype=torch.float)}

    def make_val_instance(self, args):


        if len(args) == 7:
            depot, loc, sol_static, dynamic_loc, ci_obj, mm_obj, CI_sol, *args = args
            depot = [x / 100 for x in depot]
            loc = [[x / 100 for x in inner_list] for inner_list in loc]
            dynamic_loc = [[x / 100 for x in inner_list] for inner_list in dynamic_loc]


            if len(args) > 0:
                temp = args
            return {
                'loc': torch.tensor(loc, dtype=torch.float),
                'depot': torch.tensor(depot, dtype=torch.float),
                'sol_static': torch.tensor(sol_static, dtype=torch.int),
                'dynamic_loc': torch.tensor(dynamic_loc, dtype=torch.float),
                'ci_obj': torch.tensor(ci_obj/100, dtype=torch.float),
                'mm_obj': torch.tensor(mm_obj/100, dtype=torch.float),
                'sol_MM': torch.tensor(CI_sol, dtype=torch.int)}
        elif len(args) == 6:
            depot, loc, sol_static, dynamic_loc, ci_obj, CI_sol, *args = args
            depot = [x / 100 for x in depot]
            loc = [[x / 100 for x in inner_list] for inner_list in loc]
            dynamic_loc = [[x / 100 for x in inner_list] for inner_list in dynamic_loc]


            if len(args) > 0:
                temp = args
            return {
                'loc': torch.tensor(loc, dtype=torch.float),
                'depot': torch.tensor(depot, dtype=torch.float),
                'sol_static': torch.tensor(sol_static, dtype=torch.int),
                'dynamic_loc': torch.tensor(dynamic_loc, dtype=torch.float),
                'ci_obj': torch.tensor(ci_obj/100, dtype=torch.float),
                'sol_MM': torch.tensor(CI_sol, dtype=torch.int)}
        else:
            raise ValueError("The input of the validation datasets is wrong...")

    def calculate_distance(self, data):
        N_data = data.shape[0]
        dists = torch.zeros((N_data, N_data), dtype=torch.float)
        d1 = -2 * torch.mm(data, data.T)
        d2 = torch.sum(torch.pow(data, 2), dim=1)
        d3 = torch.sum(torch.pow(data, 2), dim = 1).reshape(1,-1).T
        dists = d1 + d2 + d3
        dists[dists < 0] = 0
        return torch.sqrt(dists)
        
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.data[idx]