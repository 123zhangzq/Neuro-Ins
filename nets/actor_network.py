from torch import nn
import torch
from nets.graph_layers import AttentionEncoder_1, MultiHeadEncoder, MultiHeadDecoder, EmbeddingNet, MultiHeadPosCompat

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
    
def get_action_sig(action_record):
    action_record_tensor = torch.stack(action_record)
    return torch.cat((action_record_tensor[-3:].transpose(0,1),
      action_record_tensor.mean(0).unsqueeze(1)),1)
    
class Actor(nn.Module):

    def __init__(self,
                 problem_name,
                 embedding_dim,
                 hidden_dim,
                 n_heads_actor,
                 n_layers,
                 normalization,
                 v_range,
                 seq_length,
                 ):
        super(Actor, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads_actor = n_heads_actor
        self.n_layers = n_layers
        self.normalization = normalization
        self.range = v_range
        self.seq_length = seq_length        
        self.clac_stacks = problem_name == 'pdtspl'
        self.node_dim = 2

        # networks
        self.embedder = EmbeddingNet(
                            self.node_dim,
                            self.embedding_dim,
                            self.seq_length)
        
        self.encoder = mySequential(*(
                AttentionEncoder_1(1,
                                self.embedding_dim,
                                self.hidden_dim,
                                self.normalization,
                                )
            for _ in range(self.n_layers))) # for encoder of RoPE

        self.encoder_l2n = mySequential(*(
                MultiHeadEncoder(self.n_heads_actor,
                                self.embedding_dim,
                                self.hidden_dim,
                                self.normalization,
                                )
            for _ in range(self.n_layers))) # for the encoder of other PE

        self.pos_encoder = MultiHeadPosCompat(self.n_heads_actor, 
                                self.embedding_dim, 
                                self.hidden_dim, 
                                ) # for PFEs
        
        self.decoder = MultiHeadDecoder(input_dim = self.embedding_dim, 
                                        embed_dim = self.embedding_dim,
                                        v_range = self.range) # the two propsoed decoders
        
        print(self.get_parameter_number())

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def forward(self, problem, x_in, solution, action_his, step_info, epsilon_info = None, do_sample = False, fixed_action = None, require_entropy = False, to_critic = False, only_critic  = False):

        # the embedded input x
        bs, gs, in_d = x_in.size()
        h_embed, freqs_cis, visited_time = self.embedder(x_in, solution, step_info)
        
        # pass through encoder
        h_em = self.encoder(h_embed, freqs_cis)[0]
       # h_em = self.encoder_l2n(h_em)

        
        if only_critic:
            return (h_em)

        visited_order_map = problem.get_visited_order_map(visited_time,step_info)
        del visited_time
        
        # pass through decoder
        action, log_ll, entropy, CI_action = self.decoder(problem,
                                                h_em, 
                                                solution,
                                                action_his,
                                                step_info,
                                                x_in,
                                                visited_order_map,
                                                epsilon_info,
                                                fixed_action,
                                                require_entropy = require_entropy,
                                                do_sample = do_sample)

        if require_entropy:
            return action, log_ll.squeeze(), (h_em) if to_critic else None, entropy, CI_action
        else:
            return action, log_ll.squeeze(), (h_em) if to_critic else None, CI_action
