import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from torch import nn
import math

TYPE_REMOVAL = 'N2S'   # Neuro-Ins
#TYPE_REMOVAL = 'random'
#TYPE_REMOVAL = 'greedy'

TYPE_REINSERTION = 'N2S'    # Neuro-Ins
#TYPE_REINSERTION = 'random'
#TYPE_REINSERTION = 'greedy'


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q):
        
        h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim) #################   reshape
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)   
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        attn = F.softmax(compatibility, dim=-1)   
       
        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out

class MultiHeadAttentionNew(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttentionNew, self).__init__()

        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))
         
        self.score_aggr = nn.Sequential(
                        nn.Linear(8, 8),
                        nn.ReLU(inplace=True),
                        nn.Linear(8, 4))


        self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()
        
    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    # 旋转位置编码计算
    def apply_rotary_emb(self,
            xq: torch.Tensor,
            xk: torch.Tensor,
            freqs_cis: torch.Tensor,
    ):
        # xq.shape = [batch_size, seq_len, dim]
        # xq_.shape = [batch_size, seq_len, dim // 2, 2]
        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

        # 转为复数域
        xq_ = torch.view_as_complex(xq_)
        xk_ = torch.view_as_complex(xk_)

        # 应用旋转操作，然后将结果转回实数域
        # xq_out.shape = [batch_size, seq_len, dim]
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def forward(self, h, out_source_attn):
        
        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()

        hflat = h.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (batch_size, graph_size, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(hflat, self.W_query).view(shp)
        K = torch.matmul(hflat, self.W_key).view(shp)   
        V = torch.matmul(hflat, self.W_val).view(shp)

        # attention 操作之前，应用旋转位置编码
        Q, K = self.apply_rotary_emb(Q, K, out_source_attn)

        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(input_dim)
        scores = F.softmax(scores.float(), dim=-1)
        output = torch.matmul(scores, V)


        return output, out_source_attn

class MultiHeadPosCompat(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadPosCompat, self).__init__()
    
        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, pos):
        
        batch_size, graph_size, input_dim = pos.size()
        posflat = pos.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(posflat, self.W_query).view(shp)  
        K = torch.matmul(posflat, self.W_key).view(shp)   

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        return torch.matmul(Q, K.transpose(2, 3))

class MultiHeadCompat(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadCompat, self).__init__()
    
        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h = None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim) #################   reshape
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)  
        K = torch.matmul(hflat, self.W_key).view(shp)   

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility_s2n = torch.matmul(Q, K.transpose(2, 3))
        
        return  compatibility_s2n

class LinearSelect(nn.Module):
    def __init__(
            self,
            embed_dim
    ):
        super(LinearSelect, self).__init__()
        self.embed_dim = embed_dim
        
        # self.agg = MLP(self.embed_dim, 64, 64, 1, 0)
        self.proj = nn.Linear(self.embed_dim, 1, bias=False)

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, h):

        compatibility_pairing = self.proj(h).squeeze()
        
        return  compatibility_pairing

class Reinsertion(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(Reinsertion, self).__init__()
    
        n_heads = 4
        
        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        
        self.norm_factor = 1 / math.sqrt(2 * embed_dim)  # See Attention is all you need


        self.compater_insert1 = MultiHeadCompat(n_heads,
                                        embed_dim,
                                        embed_dim,
                                        embed_dim,
                                        key_dim)
        
        self.compater_insert2 = MultiHeadCompat(n_heads,
                                        embed_dim,
                                        embed_dim,
                                        embed_dim,
                                        key_dim)
        
        self.agg = MLP(16, 32, 32, 1, 0)

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)


    def forward(self, h, pos_pickup, pos_delivery, rec, mask=None):

        batch_size, graph_size, input_dim = h.size()
        shp = (batch_size, graph_size, graph_size, self.n_heads)
        shp_p = (batch_size, -1, 1, self.n_heads)
        shp_d = (batch_size, 1, -1, self.n_heads)
        
        arange = torch.arange(batch_size, device = h.device)
        h_pickup = h[arange,pos_pickup].unsqueeze(1)
        h_delivery = h[arange,pos_delivery].unsqueeze(1)
        h_K_neibour = h.gather(1, rec.view(batch_size, graph_size, 1).expand_as(h))

        # not return to the depot START
        mask_last_node = (rec == 0).unsqueeze(-1).expand_as(h_K_neibour)
        h_K_neibour_P = h_K_neibour.clone()
        h_K_neibour_D = h_K_neibour.clone()
        h_K_neibour_P[mask_last_node] = h_pickup.clone().expand_as(h_K_neibour)[mask_last_node]
        h_K_neibour_D[mask_last_node] = h_delivery.clone().expand_as(h_K_neibour)[mask_last_node]
        # not return to the depot END

        compatibility_pickup_pre = self.compater_insert1(h_pickup, h).permute(1,2,3,0).view(shp_p).expand(shp)
        compatibility_pickup_post = self.compater_insert2(h_pickup, h_K_neibour_P).permute(1,2,3,0).view(shp_p).expand(shp)
        compatibility_delivery_pre = self.compater_insert1(h_delivery, h).permute(1,2,3,0).view(shp_d).expand(shp)
        compatibility_delivery_post = self.compater_insert2(h_delivery, h_K_neibour_D).permute(1,2,3,0).view(shp_d).expand(shp)


        # p and d insert after the same node
        diag_indices = torch.arange(graph_size)

        compatibility_pickup_post_delivery = self.compater_insert2(h_pickup, h_delivery).permute(1, 2, 3, 0).view(shp_p).expand(shp)
        #compatibility_delivery_pre_pickup = self.compater_insert1(h_delivery, h_pickup).permute(1, 2, 3, 0).view(shp_d).expand(shp)
        compatibility_delivery_pre_pickup = torch.zeros(shp, device=compatibility_pickup_post_delivery.device)

        compatibility_same_node = self.agg(torch.cat((compatibility_pickup_pre,
                                            compatibility_pickup_post_delivery,
                                            compatibility_delivery_pre_pickup,
                                            compatibility_delivery_post),-1)).squeeze()


        compatibility = self.agg(torch.cat((compatibility_pickup_pre, 
                                            compatibility_pickup_post, 
                                            compatibility_delivery_pre, 
                                            compatibility_delivery_post),-1)).squeeze()

        compatibility[:, diag_indices, diag_indices] = compatibility_same_node[:, diag_indices, diag_indices]

        return compatibility

class MLP(torch.nn.Module):
    def __init__(self,
                input_dim = 128,
                feed_forward_dim = 64,
                embedding_dim = 64,
                output_dim = 1,
                p_dropout = 0.01
    ):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, feed_forward_dim)
        self.fc2 = torch.nn.Linear(feed_forward_dim, embedding_dim)
        self.fc3 = torch.nn.Linear(embedding_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=p_dropout)
        self.ReLU = nn.ReLU(inplace = True)
        
        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, in_):
        result = self.ReLU(self.fc1(in_))
        result = self.dropout(result)
        result = self.ReLU(self.fc2(result))
        result = self.fc3(result).squeeze(-1)
        return result

class ValueDecoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            input_dim,
    ):
        super(ValueDecoder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        self.project_graph = nn.Linear(self.input_dim, self.embed_dim // 2)
        
        self.project_node = nn.Linear(self.input_dim, self.embed_dim // 2) 
        
        self.MLP = MLP(input_dim + 1, embed_dim)


    def forward(self, h_em, cost): 
                
        # get embed feature
#        max_pooling = h_em.max(1)[0]   # max Pooling
        mean_pooling = h_em.mean(1)     # mean Pooling
        graph_feature = self.project_graph(mean_pooling)[:, None, :]
        node_feature = self.project_node(h_em)
        
        #pass through value_head, get estimated value
        fusion = node_feature + graph_feature.expand_as(node_feature) # torch.Size([2, 50, 128])

        fusion_feature = torch.cat((fusion.mean(1),
                                    fusion.max(1)[0],
                                    cost.to(h_em.device),
                                    ), -1)
        
        value = self.MLP(fusion_feature)
      
        return value



class MultiHeadDecoder(nn.Module):
    def __init__(
            self,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None,
            v_range = 6,
    ):
        super(MultiHeadDecoder, self).__init__()
        self.n_heads = n_heads = 1
        self.embed_dim = embed_dim
        self.input_dim = input_dim        
        self.range = v_range
        
        if TYPE_REMOVAL == 'N2S':
            self.select_order = LinearSelect(embed_dim)
        if TYPE_REINSERTION == 'N2S':
            self.compater_reinsertion = Reinsertion(n_heads,
                                            embed_dim,
                                            embed_dim,
                                            embed_dim,
                                            key_dim)
            
        self.project_graph = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.project_node = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
        
        
    def forward(self, problem, h_em, solutions, action_his, step_info, x_in, visited_order_map, epsilon_info = None, fixed_action = None, require_entropy = False, do_sample = True):
        # size info
        dy_size, dy_t = step_info

        bs, gs, dim = h_em.size()
        dy_half_pos =  dy_size // 2
        dy_pos = gs - dy_size + dy_t

        arange = torch.arange(bs)

        # w/ or w/o graph embedding
        # h = self.project_node(h_em) + self.project_graph(h_em.max(1)[0])[:, None, :].expand(bs, gs, dim)
        h = h_em
        

        ############# action1 select a dynamic order
        if TYPE_REMOVAL == 'N2S':
            action_removal_table = torch.tanh(self.select_order(h).squeeze()) * self.range

            # mask the other nodes apart from candidates
            dy_delivery = int(gs - dy_size + dy_size / 2)

            action_removal_table[arange, :int(gs - dy_size)] = -1e20
            action_removal_table[arange, dy_delivery:] = -1e20

            mask_selected = torch.zeros_like(action_removal_table, dtype=torch.bool, device=action_removal_table.device)
            mask_selected[solutions != 0] = True
            action_removal_table[mask_selected] = -1e20

            log_ll_removal = F.log_softmax(action_removal_table, dim = -1) if self.training and TYPE_REMOVAL == 'N2S' else None
            probs_removal = F.softmax(action_removal_table, dim = -1)
        elif TYPE_REMOVAL == 'random':
            probs_removal = torch.rand(bs, gs//2).to(h_em.device)
        else:
            pass

        if fixed_action is not None:
            action_removal = fixed_action[:,:1]
        else:
            if TYPE_REMOVAL == 'random':
                action_removal = torch.full((bs, 1), fill_value=dy_pos, dtype=torch.long).to(h_em.device)
            else:
                if do_sample:
                    action_removal = probs_removal.multinomial(1)
                else:
                    action_removal = probs_removal.max(-1)[1].unsqueeze(1)


        selected_log_ll_action1 = log_ll_removal.gather(1, action_removal) if self.training and TYPE_REMOVAL == 'N2S' else torch.tensor(0).to(h.device)

        if action_his is not None:
            action_his.scatter_(1, action_removal, True)
            action_his.scatter_(1, action_removal + dy_half_pos, True)

        ############# action2 insert into current routes
        pos_pickup = action_removal.view(-1)
        pos_delivery = pos_pickup + dy_half_pos
        mask_table = problem.get_swap_mask(action_removal, visited_order_map, step_info, action_his).expand(bs, gs, gs).cpu()
        if TYPE_REINSERTION == 'N2S':
            action_reinsertion_table = torch.tanh(self.compater_reinsertion(h, pos_pickup, pos_delivery, solutions, mask_table)) * self.range
        elif TYPE_REINSERTION == 'random':
            action_reinsertion_table = torch.ones(bs, gs, gs).to(h_em.device)
        else:
            # epi-greedy
            pos_pickup = action_removal
            pos_delivery = pos_pickup + dy_half_pos
            rec_new = solutions.clone()

            # perform calc on new rec_new
            first_row = torch.arange(gs, device = solutions.device).long().unsqueeze(0).expand(bs, gs)
            d_i =  x_in.gather(1, first_row.unsqueeze(-1).expand(bs, gs, 2))
            d_i_next = x_in.gather(1, rec_new.long().unsqueeze(-1).expand(bs, gs, 2))
            d_pick = x_in.gather(1, pos_pickup.unsqueeze(1).expand(bs, gs, 2))
            d_deli = x_in.gather(1, pos_delivery.unsqueeze(1).expand(bs, gs, 2))
            cost_insert_p = (d_pick  - d_i).norm(p=2, dim=2) + (d_pick  - d_i_next).norm(p=2, dim=2) - (d_i  - d_i_next).norm(p=2, dim=2)
            cost_insert_d = (d_deli  - d_i).norm(p=2, dim=2) + (d_deli  - d_i_next).norm(p=2, dim=2) - (d_i  - d_i_next).norm(p=2, dim=2)

            # not to return depot
            zero_indices = (solutions == 0).to(torch.bool)
            cost_insert_p[zero_indices] = (d_pick  - d_i).norm(p=2, dim=2)[zero_indices]
            cost_insert_d[zero_indices] = (d_deli  - d_i).norm(p=2, dim=2)[zero_indices]
            action_reinsertion_table = - (cost_insert_p.view(bs, gs, 1) + cost_insert_d.view(bs, 1, gs))
            ######################## above is the CI#######################

            action_reinsertion_table_random = torch.ones(bs, gs, gs).to(h_em.device)
            action_reinsertion_table_random[mask_table] = -1e20
            action_reinsertion_table_random = action_reinsertion_table_random.view(bs, -1)
            probs_reinsertion_random = F.softmax(action_reinsertion_table_random, dim = -1)
             
        action_reinsertion_table[mask_table] = -1e20


        #reshape action_reinsertion_table
        action_reinsertion_table = action_reinsertion_table.view(bs, -1)
        log_ll_reinsertion = F.log_softmax(action_reinsertion_table, dim = -1) if self.training and TYPE_REINSERTION == 'N2S' else None
        probs_reinsertion = F.softmax(action_reinsertion_table, dim = -1)

        # fixed action
        if fixed_action is not None:
            p_selected = fixed_action[:,1]
            d_selected = fixed_action[:,2]
            pair_index = p_selected * gs + d_selected
            pair_index = pair_index.view(-1,1)
            action = fixed_action
        else:
            if TYPE_REINSERTION == 'greedy':
                action_reinsertion_random = probs_reinsertion_random.multinomial(1)
                action_reinsertion_greedy = probs_reinsertion.max(-1)[1].unsqueeze(1)
                # pair_index = torch.where(torch.rand(bs,1).to(h_em.device) < 0.1, action_reinsertion_random, action_reinsertion_greedy)
                pair_index = action_reinsertion_greedy
            elif not do_sample:
                action_reinsertion_greedy = probs_reinsertion.max(-1)[1].unsqueeze(1)
                pair_index = action_reinsertion_greedy
            else:
                # # pure sample one action
                # pair_index = probs_reinsertion.multinomial(1)

                # e-greedy sample one action
                epsilon, epsilon_decay, epoch = epsilon_info
                epoch -= 1000
                epsilon = epsilon * np.exp(-epsilon_decay * epoch)
                action_reinsertion_sample = probs_reinsertion.multinomial(1)
                action_reinsertion_greedy = probs_reinsertion.max(-1)[1].unsqueeze(1)
                pair_index = torch.where(torch.rand(bs, 1).to(h_em.device) < epsilon, action_reinsertion_sample,
                                         action_reinsertion_greedy)
            
            p_selected = pair_index // gs
            d_selected = pair_index % gs
            action = torch.cat((action_removal.view(bs, -1), p_selected, d_selected),-1)  # pair: no_head bs, 2
        
        selected_log_ll_action2 = log_ll_reinsertion.gather(1, pair_index)  if self.training and TYPE_REINSERTION == 'N2S' else torch.zeros((bs, 1)).to(h.device)
        
        # log_ll = selected_log_ll_action1 + selected_log_ll_action2
        log_ll = selected_log_ll_action2 + selected_log_ll_action1
        
        if require_entropy and self.training:
            dist = Categorical(probs_reinsertion, validate_args=False)
            entropy = dist.entropy()
        else:
            entropy = None


        # action of CI
        pos_pickup = action_removal
        pos_delivery = pos_pickup + dy_half_pos
        rec_new = solutions.clone()
        first_row = torch.arange(gs, device=solutions.device).long().unsqueeze(0).expand(bs, gs)
        d_i = x_in.gather(1, first_row.unsqueeze(-1).expand(bs, gs, 2))
        d_i_next = x_in.gather(1, rec_new.long().unsqueeze(-1).expand(bs, gs, 2))
        d_pick = x_in.gather(1, pos_pickup.unsqueeze(1).expand(bs, gs, 2))
        d_deli = x_in.gather(1, pos_delivery.unsqueeze(1).expand(bs, gs, 2))
        cost_insert_p = (d_pick - d_i).norm(p=2, dim=2) + (d_pick - d_i_next).norm(p=2, dim=2) - (d_i - d_i_next).norm(
            p=2, dim=2)
        cost_insert_d = (d_deli - d_i).norm(p=2, dim=2) + (d_deli - d_i_next).norm(p=2, dim=2) - (d_i - d_i_next).norm(
            p=2, dim=2)
        # not to return depot
        zero_indices = (solutions == 0).to(torch.bool)
        cost_insert_p[zero_indices] = (d_pick - d_i).norm(p=2, dim=2)[zero_indices]
        cost_insert_d[zero_indices] = (d_deli - d_i).norm(p=2, dim=2)[zero_indices]
        action_reinsertion_table = - (cost_insert_p.view(bs, gs, 1) + cost_insert_d.view(bs, 1, gs))
        # p and d insert after the same node
        cost_insert_same_node = -((d_pick - d_i).norm(p=2, dim=2) + (d_pick - d_deli).norm(p=2, dim=2) + \
                                  (d_deli - d_i_next).norm(p=2, dim=2) - (d_i - d_i_next).norm(p=2, dim=2))
        cost_insert_same_node[zero_indices] = (-(d_pick - d_i).norm(p=2, dim=2) - (d_pick - d_deli).norm(p=2, dim=2))[
            zero_indices]
        action_reinsertion_table.diagonal(dim1=-2, dim2=-1).zero_()
        diagonal_matrix = torch.diag_embed(cost_insert_same_node)
        action_reinsertion_table += diagonal_matrix
        action_reinsertion_table[mask_table] = -1e20
        action_reinsertion_table = action_reinsertion_table.view(bs, -1)
        probs_reinsertion = F.softmax(action_reinsertion_table, dim = -1)
        action_reinsertion_greedy = probs_reinsertion.max(-1)[1].unsqueeze(1)
        pair_index = action_reinsertion_greedy
        p_selected_GI = pair_index // gs
        d_selected_GI = pair_index % gs
        GI_action = torch.cat((action_removal.view(bs, -1), p_selected_GI, d_selected_GI), -1)  # pair: no_head bs, 2
        del visited_order_map, mask_table


        return action, log_ll, entropy, GI_action


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalization = normalization

        if not self.normalization == 'layer':
            self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.normalization == 'layer':
            return (input - input.mean((1,2)).view(-1,1,1)) / torch.sqrt(input.var((1,2)).view(-1,1,1) + 1e-05)

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input

class AttentionEncoder_1(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(AttentionEncoder_1, self).__init__()

        self.MHA_sublayer = MultiHeadAttentionsubLayer_1(
                        n_heads,
                        embed_dim,
                        feed_forward_hidden,
                        normalization=normalization,
                )
        
        self.FFandNorm_sublayer = FFandNormsubLayer(
                        n_heads,
                        embed_dim,
                        feed_forward_hidden,
                        normalization=normalization,
                )
        
    def forward(self, input1, input2):
        out1, out2 = self.MHA_sublayer(input1, input2)
        return self.FFandNorm_sublayer(out1), out2


class MultiHeadEncoder(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(MultiHeadEncoder, self).__init__()

        self.MHA_sublayer = MultiHeadAttentionsubLayer(
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization=normalization,
        )

        self.FFandNorm_sublayer = FFandNormsubLayer(
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization=normalization,
        )

    def forward(self, input):
        out = self.MHA_sublayer(input)
        return self.FFandNorm_sublayer(out)
    
class MultiHeadAttentionsubLayer_1(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(MultiHeadAttentionsubLayer_1, self).__init__()
        
        self.MHA = MultiHeadAttentionNew(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
        
        self.Norm = Normalization(embed_dim, normalization)
    
    def forward(self, input1, input2):
        # Attention and Residual connection
        out1, out2 = self.MHA(input1, input2)
        
        # Normalization
        return self.Norm(out1 + input1), out2


class MultiHeadAttentionsubLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(MultiHeadAttentionsubLayer, self).__init__()

        self.MHA = MultiHeadAttention(
            n_heads,
            input_dim=embed_dim,
            embed_dim=embed_dim
        )

        self.Norm = Normalization(embed_dim, normalization)

    def forward(self, input):
        # Attention and Residual connection
        out = self.MHA(input)

        # Normalization
        return self.Norm(out + input)
   
class FFandNormsubLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(FFandNormsubLayer, self).__init__()
        
        self.FF = nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden, bias = False),
                    nn.ReLU(inplace = True),
                    nn.Linear(feed_forward_hidden, embed_dim, bias = False)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim, bias = False)
        
        self.Norm = Normalization(embed_dim, normalization)
        
    
    def forward(self, input):
    
        # FF and Residual connection
        out = self.FF(input)
        # Normalization
        return self.Norm(out + input)


class EmbeddingNet(nn.Module):
    
    def __init__(
            self,
            node_dim,
            embedding_dim,
            seq_length,
        ):
        super(EmbeddingNet, self).__init__()
        self.node_dim = node_dim
        self.embedding_dim = embedding_dim
        self.embedder = nn.Linear(node_dim, embedding_dim, bias = False)

        # self.pattern = self.cyclic_position_encoding_pattern(2 * seq_length, embedding_dim)
        self.seq_length = seq_length

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)


    def get_visited_time(self, solutions, step_info):
        # size info
        dy_size, dy_t = step_info
        batch_size, seq_length = solutions.size()

        valid_seq_length = seq_length - dy_size + 2 * dy_t
        valid_half_size = valid_seq_length // 2  # TODO: need to change to the dynamic version, includes the two clac_stacks below


        # get index according to the solutions
        visited_time = torch.zeros((batch_size, seq_length), device=solutions.device)

        pre = torch.zeros((batch_size), device=solutions.device).long()

        arange = torch.arange(batch_size)

        for i in range(valid_seq_length):

            # calculate visited_time
            current_nodes = solutions[arange, pre]
            visited_time[arange, current_nodes] = i + 1
            pre = solutions[arange, pre]

        index = (visited_time % valid_seq_length).long()
        return index, visited_time.long()

    def precompute_freqs_cis(self, dim: int, index, theta: float = 10000.0):
        # t = [0, 1,..., seq_len-1]
        t = index.to(torch.float)

        # The rotation angle assigned to each pair after grouping the embedding dimensions two by two
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        freqs = freqs.unsqueeze(0).expand(t.size(0), -1).to(index.device)

        # freqs.shape = [seq_len, dim // 2]
        freqs = torch.matmul(t.unsqueeze(2), freqs.unsqueeze(1))   # 计算m * \theta

        # complex vector calculation
        # if freqs = [x, y]
        # then freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis
        
    def forward(self, x, solutions, step_info):
        index_for_freqs, visited_time = self.get_visited_time(solutions, step_info)
        freqs_cis = self.precompute_freqs_cis(self.embedding_dim, index_for_freqs)

        x_embedding = self.embedder(x)
        return  x_embedding, freqs_cis, visited_time
    
class MultiHeadAttentionLayerforCritic(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(MultiHeadAttentionLayerforCritic, self).__init__(
            SkipConnection(
                    MultiHeadAttention(
                        n_heads,
                        input_dim=embed_dim,
                        embed_dim=embed_dim
                    )                
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                    nn.Sequential(
                        nn.Linear(embed_dim, feed_forward_hidden),
                        nn.ReLU(inplace = True),
                        nn.Linear(feed_forward_hidden, embed_dim,)
                    ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        ) 
