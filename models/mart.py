import torch
import torch.nn as nn

import numpy as np

from .prtuncertain import RT, RTNoEdgeInit
from .hrtuncertain import HRT, HRTNoEdgeInit

import torch.nn.functional as F
import math

def interpolate_between_past_future(
    past: torch.Tensor,
    future: torch.Tensor,
    num_interp: int = 10,
    future_index: int = 0,
) -> torch.Tensor:
    """
    在 past 的最后一帧和 future 的指定帧之间做线性插值。

    Args:
        past: Tensor, shape (N, T_past, C)
        future: Tensor, shape (N, T_future, C)
        num_interp: 需要生成的插值点数量
        future_index: 选取 future 的哪一帧作为终点，默认第 0 帧

    Returns:
        interp: Tensor, shape (N, num_interp, C)

    Notes:
        - 不包含 past 的最后一帧
        - 包含 future[:, future_index, :]
    """
    if past.ndim != 3 or future.ndim != 3:
        raise ValueError(f"past 和 future 必须是 3 维张量，当前 past={past.shape}, future={future.shape}")

    if past.shape[0] != future.shape[0]:
        raise ValueError(f"batch 维度不一致: past.shape[0]={past.shape[0]}, future.shape[0]={future.shape[0]}")

    if past.shape[2] != future.shape[2]:
        raise ValueError(f"特征维度不一致: past.shape[2]={past.shape[2]}, future.shape[2]={future.shape[2]}")

    if not (0 <= future_index < future.shape[1]):
        raise IndexError(f"future_index={future_index} 超出范围，future.shape[1]={future.shape[1]}")

    if num_interp <= 0:
        raise ValueError(f"num_interp 必须大于 0，当前为 {num_interp}")

    start = past[:, -1:, :]                         # (N, 1, C)
    end = future[:, future_index:future_index+1, :] # (N, 1, C)

    # 例如 num_interp=10:
    # alpha = [0.1, 0.2, ..., 1.0]
    alpha = torch.linspace(
        1.0 / num_interp,
        1.0,
        steps=num_interp,
        device=past.device,
        dtype=past.dtype
    ).view(1, num_interp, 1)  # (1, num_interp, 1)

    interp = start + alpha * (end - start)         # (N, num_interp, C)
    return interp

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(1024, 512), activation='relu'):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_dims)
        dims.append(output_dim)
        layers = []
        self.out_dim = hidden_dims[-1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU(inplace=True))
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class PositionalAgentEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_t_len=200, concat=True):
        super(PositionalAgentEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        self.d_model = d_model
        if concat:
            self.fc = nn.Linear(2 * d_model, d_model)

        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def get_pos_enc(self, num_t, num_a, t_offset):
        pe = self.pe[t_offset: num_t + t_offset, :]
        pe = pe[None].repeat(num_a, 1, 1)
        return pe

    def get_agent_enc(self, num_t, num_a, a_offset):
        ae = self.ae[a_offset: num_a + a_offset, :]
        ae = ae.repeat(num_t, 1, 1)
        return ae

    def forward(self, x, num_a, t_offset=0):
        num_t = x.shape[1]
        pos_enc = self.get_pos_enc(num_t, num_a, t_offset) #(N, T, D)
        if self.concat:
            feat = [x, pos_enc]
            x = torch.cat(feat, dim=-1)
            x = self.fc(x)
        else:
            x += pos_enc
        return self.dropout(x) #(N, T, D)


# class Decoder(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.multiplier = len(args.hyper_scales) + 1
#         self.decoder_mlp = MLP(
#             args.model_dim*self.multiplier,
#             args.future_length*2,
#             hidden_dims=(
#                 args.decoder_hidden_dim,
#                 args.decoder_hidden_dim // 2
#             )
#         )

#     def forward(self, final_feature, cur_location):
#         outputs = self.decoder_mlp(final_feature)
#         outputs = outputs.view(-1, self.args.future_length, 2)
#         if not self.args.pred_rel:
#             outputs = outputs + cur_location
        
#         return outputs        

class ResidualDecoderBlock(nn.Module):

    def __init__(
        self,
        hidden_dim,
        dropout=0.0
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.norm = nn.LayerNorm(
            hidden_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x
    ):
        residual = x

        out = self.fc1(
            x
        )

        out = F.relu(
            out,
            inplace=True
        )

        out = self.dropout(
            out
        )

        out = self.fc2(
            out
        )

        out = self.dropout(
            out
        )

        out = self.norm(
            residual + out
        )

        out = F.relu(
            out,
            inplace=True
        )

        return out



class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        H = getattr(args, "decoder_hidden_dim", args.hidden_dim)

        self.feature_fc = nn.Linear(args.model_dim * 3, H)
        self.unc_film = nn.Sequential(nn.Linear(1, H // 2), nn.ReLU(inplace=True), nn.Linear(H // 2, H * 2))
        self.decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(H, H // 2),
            nn.ReLU(inplace=True),
            nn.Linear(H // 2, args.future_length * 2)
        )

    def forward(self, feature, cur_pos, uncertainty):
        h = self.feature_fc(feature)
        gamma, beta = self.unc_film(uncertainty).chunk(2, dim=-1)
        gamma, beta = 0.1 * torch.tanh(gamma), 0.1 * torch.tanh(beta)
        h = h * (1.0 + gamma) + beta
        out = self.decoder(h).view(-1, self.args.future_length, 2)
        if not self.args.pred_rel: out = out + cur_pos
        return out

    
# class Decoder(nn.Module):

#     def __init__(
#         self,
#         args
#     ):
#         super().__init__()

#         self.args = args

#         # ==================================================
#         # Keep exactly the same input dimension
#         # as the original MART decoder
#         # ==================================================

#         self.multiplier = (
#             len(args.hyper_scales)
#             + 1
#         )

#         input_dim = (
#             args.model_dim
#             * self.multiplier
#         )

#         # ==================================================
#         # Capacity-matched hidden dimension
#         #
#         # Current model:
#         # ~0.99M
#         #
#         # Original MART:
#         # ~1.58M
#         #
#         # Need roughly +0.59M parameters.
#         #
#         # 448 is a good first choice.
#         # ==================================================

#         self.hidden_dim = getattr(
#             args,
#             "decoder_matched_hidden_dim",
#             448
#         )

#         dropout = getattr(
#             args,
#             "decoder_dropout",
#             0.0
#         )

#         # ==================================================
#         # Input projection
#         #
#         # final_feature
#         # -> strong latent representation
#         # ==================================================

#         self.input_proj = nn.Sequential(

#             nn.Linear(
#                 input_dim,
#                 self.hidden_dim
#             ),

#             nn.LayerNorm(
#                 self.hidden_dim
#             ),

#             nn.ReLU(
#                 inplace=True
#             )
#         )

#         # ==================================================
#         # Residual nonlinear decoding
#         #
#         # Important:
#         # sampled interaction differences are allowed
#         # to generate nonlinear trajectory differences.
#         # ==================================================

#         self.res_block = (
#             ResidualDecoderBlock(
#                 hidden_dim=
#                     self.hidden_dim,

#                 dropout=
#                     dropout
#             )
#         )

#         # ==================================================
#         # Trajectory head
#         # ==================================================

#         self.output_head = nn.Sequential(

#             nn.Linear(
#                 self.hidden_dim,
#                 self.hidden_dim // 2
#             ),

#             nn.ReLU(
#                 inplace=True
#             ),

#             nn.Linear(
#                 self.hidden_dim // 2,
#                 args.future_length * 2
#             )
#         )

    # def forward(
    #     self,
    #     final_feature,
    #     cur_location
    # ):
    #     """
    #     final_feature:
    #         [B*N, D]
    #         or any shape supported by Linear
    #         with last dim = input_dim

    #     output:
    #         [-1, T_future, 2]
    #     """

    #     # ==================================================
    #     # Strong feature projection
    #     # ==================================================

    #     hidden = self.input_proj(
    #         final_feature
    #     )

    #     # ==================================================
    #     # Residual nonlinear mapping
    #     # ==================================================

    #     hidden = self.res_block(
    #         hidden
    #     )

    #     # ==================================================
    #     # Decode full future trajectory
    #     # ==================================================

    #     outputs = self.output_head(
    #         hidden
    #     )

    #     outputs = outputs.view(
    #         -1,
    #         self.args.future_length,
    #         2
    #     )

    #     # ==================================================
    #     # Absolute prediction
    #     # ==================================================

    #     if not self.args.pred_rel:

    #         outputs = (
    #             outputs
    #             + cur_location
    #         )

    #     return outputs

    
class InteractionConditionedDecoder(nn.Module):

    def __init__(
        self,
        args
    ):
        super().__init__()

        self.args = args

        D = args.model_dim

        hidden_dim = getattr(
            args,
            "decoder_hidden_dim",
            512
        )

        # ==============================================
        # Base:
        #
        # individual + group
        # ==============================================

        self.base_encoder = nn.Sequential(
            nn.Linear(
                2 * D,
                hidden_dim
            ),
            nn.ReLU(inplace=True),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.ReLU(inplace=True)
        )

        # ==============================================
        # Interaction conditioner
        #
        # n_pair_s -> gamma, beta
        # ==============================================

        self.interaction_conditioner = nn.Sequential(
            nn.Linear(
                D,
                hidden_dim
            ),
            nn.ReLU(inplace=True),

            nn.Linear(
                hidden_dim,
                2 * hidden_dim
            )
        )

        # ==============================================
        # Refinement
        # ==============================================

        self.refine = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.ReLU(inplace=True),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.ReLU(inplace=True)
        )

        # ==============================================
        # Trajectory head
        # ==============================================

        self.traj_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim // 2
            ),
            nn.ReLU(inplace=True),

            nn.Linear(
                hidden_dim // 2,
                args.future_length * 2
            )
        )

    def forward(
        self,
        n_initial,
        n_pair,
        n_group,
        cur_location
    ):

        # ==============================================
        # Base representation
        # ==============================================

        base_feature = torch.cat(
            [
                n_initial,
                n_group
            ],
            dim=-1
        )

        base_feature = self.base_encoder(
            base_feature
        )

        # ==============================================
        # Interaction-conditioned FiLM
        # ==============================================

        gamma_beta = (
            self.interaction_conditioner(
                n_pair
            )
        )

        gamma, beta = (
            gamma_beta.chunk(
                2,
                dim=-1
            )
        )

        gamma = (
            0.1
            * torch.tanh(
                gamma
            )
        )

        beta = (
            0.1
            * torch.tanh(
                beta
            )
        )

        feature = (
            base_feature
            * (1.0 + gamma)
            + beta
        )

        # ==============================================
        # Refinement
        # ==============================================

        feature = (
            feature
            + self.refine(
                feature
            )
        )

        # ==============================================
        # Trajectory
        # ==============================================

        outputs = (
            self.traj_head(
                feature
            )
        )

        outputs = outputs.view(
            -1,
            self.args.future_length,
            2
        )

        if not self.args.pred_rel:

            outputs = (
                outputs
                + cur_location
            )

        return outputs


class MART(nn.Module):
    def __init__(self, args):
        super(MART, self).__init__()
        self.args = args

        module_args = {
            'num_layers': 1,
            'num_heads': args.num_heads,
            'node_dim': args.model_dim,
            'node_hidden_dim': args.hidden_dim,
            'edge_dim': args.model_dim,
            'edge_hidden_dim_1': args.hidden_dim,
            'edge_hidden_dim_2': args.hidden_dim,
            'dropout': args.dropout,
        }
        
        self.input_dim = len(args.inputs)
        self.input_fc = nn.Linear(self.input_dim, args.model_dim)
        self.input_fc2 = nn.Linear(args.model_dim*args.past_length, args.model_dim)
        
        self.pos_encoder = PositionalAgentEncoding(args.model_dim, 0.1, concat=True)
        
        self.pair_encoders = nn.ModuleList()
        self.hyper_encoders = nn.ModuleList()
        
        for i in range(args.num_layers):
            if i == 0:
                self.pair_encoders.append(RT(**module_args))
            else:
                self.pair_encoders.append(RTNoEdgeInit(**module_args))
        
        module_args['function_type'] = args.function_type
        
        for i in range(args.num_layers):
            if i == 0:
                self.hyper_encoders.append(HRT(**module_args))
            else:
                self.hyper_encoders.append(HRTNoEdgeInit(**module_args))
        
        for i in range(args.sample_k):
            self.add_module("head_%d" % i, Decoder(args))
        
    def forward(self, x_abs, x_rel):
        inputs = []
        batch_size, num_agents, length, _ = x_abs.shape
        cur_pos = x_abs[:, :, [-1]].view(batch_size*num_agents, 1, -1).contiguous()
                
        if 'pos_x' in self.args.inputs and 'pos_y' in self.args.inputs:
            inputs.append(x_abs)
        if 'vel_x' in self.args.inputs and 'vel_y' in self.args.inputs:
            inputs.append(x_rel)
        
        inputs = torch.cat(inputs, dim=-1)
        inputs = inputs.view(batch_size*num_agents, length, -1).contiguous()
        
        inputs_fc = self.input_fc(inputs).view(batch_size*num_agents, length, self.args.model_dim)
        inputs_pos = self.pos_encoder(inputs_fc, num_a=batch_size*num_agents)
        inputs_pos = inputs_pos.view(batch_size, num_agents, length, self.args.model_dim)
        n_initial = self.input_fc2(inputs_pos.contiguous().view(batch_size, num_agents, length*self.args.model_dim))
        
        n_pair, e_pair = n_initial, None
        n_group, e_group, G = n_initial, None, None
        
        for i in range(self.args.num_layers):
            n_pair, e_pair = self.pair_encoders[i](n_pair, e_pair, return_edge=True)
            n_group, e_group, G = self.hyper_encoders[i](n_group, e_group, G, return_edge=True)
        
        n_final = torch.cat([n_initial, n_pair, n_group], dim=-1)
        
        out_list = []
        for i in range(self.args.sample_k):
            out = self._modules["head_%d" % i](n_final, cur_pos)
            out_list.append(out[:, None, :, :])
        
        out = torch.cat(out_list, dim=2)
        out = out.view(batch_size, num_agents, self.args.sample_k, self.args.future_length, -1)
        
        return out
    

class CoarseEncoderMLP(nn.Module):
    def __init__(self, args, future=False):
        super().__init__()
        self.args = args

        self.model_dim = 192


        self.input_dim = len(args.inputs)
        self.input_fc = nn.Linear(self.input_dim, self.model_dim)
        if future:
            self.input_fc2 = nn.Linear(self.model_dim*args.future_length, self.model_dim)
        else:
            self.input_fc2 = nn.Linear(self.model_dim*args.past_length, self.model_dim)
        
        self.pos_encoder = PositionalAgentEncoding(self.model_dim, 0.1, concat=True)

    def forward(self, x_abs, x_rel, offset=0):
        inputs = []
        batch_size, num_agents, length, _ = x_abs.shape
        cur_pos = x_abs[:, :, [-1]].view(batch_size*num_agents, 1, -1).contiguous()
                
        if 'pos_x' in self.args.inputs and 'pos_y' in self.args.inputs:
            inputs.append(x_abs)
        if 'vel_x' in self.args.inputs and 'vel_y' in self.args.inputs:
            inputs.append(x_rel)
        
        inputs = torch.cat(inputs, dim=-1)
        inputs = inputs.view(batch_size*num_agents, length, -1).contiguous()
        
        inputs_fc = self.input_fc(inputs).view(batch_size*num_agents, length, self.model_dim)
        inputs_pos = self.pos_encoder(inputs_fc, num_a=batch_size*num_agents, t_offset=offset)
        inputs_pos = inputs_pos.view(batch_size, num_agents, length, self.model_dim)
        n_initial = self.input_fc2(inputs_pos.contiguous().view(batch_size, num_agents, length*self.model_dim))
        
        return n_initial.view(batch_size*num_agents, -1)

class MARTModule(nn.Module):
    def __init__(self, args, future = False):
        super(MARTModule, self).__init__()
        self.args = args

        module_args = {
            'num_layers': 1,
            'num_heads': args.num_heads,
            'node_dim': args.model_dim,
            'node_hidden_dim': args.hidden_dim,
            'edge_dim': args.model_dim,
            'edge_hidden_dim_1': args.hidden_dim,
            'edge_hidden_dim_2': args.hidden_dim,
            'dropout': args.dropout,
        }

        
        self.input_dim = len(args.inputs)
        self.input_fc = nn.Linear(self.input_dim, args.model_dim)
        if future:
            self.input_fc2 = nn.Linear(args.model_dim*args.future_length, args.model_dim)
        else:
            self.input_fc2 = nn.Linear(args.model_dim*args.past_length, args.model_dim)
        
        self.pos_encoder = PositionalAgentEncoding(args.model_dim, 0.1, concat=True)
        
        self.pair_encoders = nn.ModuleList()
        self.hyper_encoders = nn.ModuleList()

        for i in range(args.num_layers):
            if i == 0:
                self.pair_encoders.append(RT(**module_args))
            else:
                self.pair_encoders.append(RTNoEdgeInit(**module_args))
        
        module_args['function_type'] = args.function_type
        
        for i in range(args.num_layers):
            if i == 0:
                self.hyper_encoders.append(HRT(**module_args))
            else:
                self.hyper_encoders.append(HRTNoEdgeInit(**module_args))
        
        for i in range(args.sample_k):
            self.add_module("head_%d" % i, Decoder(args))
        
    def forward(self, x_abs, x_rel, offset=0):
        inputs = []
        batch_size, num_agents, length, _ = x_abs.shape
        cur_pos = x_abs[:, :, [-1]].view(batch_size*num_agents, 1, -1).contiguous()
                
        if 'pos_x' in self.args.inputs and 'pos_y' in self.args.inputs:
            inputs.append(x_abs)
        if 'vel_x' in self.args.inputs and 'vel_y' in self.args.inputs:
            inputs.append(x_rel)
        
        inputs = torch.cat(inputs, dim=-1)
        inputs = inputs.view(batch_size*num_agents, length, -1).contiguous()
        
        inputs_fc = self.input_fc(inputs).view(batch_size*num_agents, length, self.args.model_dim)
        inputs_pos = self.pos_encoder(inputs_fc, num_a=batch_size*num_agents, t_offset=offset)
        inputs_pos = inputs_pos.view(batch_size, num_agents, length, self.args.model_dim)
        n_initial = self.input_fc2(inputs_pos.contiguous().view(batch_size, num_agents, length*self.args.model_dim))
        
        n_pair, e_pair = n_initial, None
        n_group, e_group, G = n_initial, None, None
        
        for i in range(self.args.num_layers):
            n_pair, e_pair = self.pair_encoders[i](n_pair, e_pair, return_edge=True)
            n_group, e_group, G = self.hyper_encoders[i](n_group, e_group, G, return_edge=True)
        
        n_final = torch.cat([n_initial, n_pair, n_group], dim=-1)
        
        # out_list = []
        # for i in range(self.args.sample_k):
        #     out = self._modules["head_%d" % i](n_final, cur_pos)
        #     out_list.append(out[:, None, :, :])
        
        # out = torch.cat(out_list, dim=2)
        # out = out.view(batch_size, num_agents, self.args.sample_k, self.args.future_length, -1)
        
        # return n_initial, n_pair, n_group
        return n_final.view(batch_size*num_agents, -1), n_initial.view(batch_size*num_agents, -1), n_pair.view(batch_size*num_agents, -1), n_group.view(batch_size*num_agents, -1)
    
def initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

class Normal:
    def __init__(self, mu=None, logvar=None, params=None):
        super().__init__()
        if params is not None:
            self.mu, self.logvar = torch.chunk(params, chunks=2, dim=-1)
        else:
            assert mu is not None
            assert logvar is not None
            self.mu = mu
            self.logvar = logvar
        self.sigma = torch.exp(0.5 * self.logvar)

    def rsample(self):
        eps = torch.randn_like(self.sigma)
        return self.mu + eps * self.sigma

    def sample(self):
        return self.rsample()

    def kl(self, p=None):
        """ compute KL(q||p) """
        if p is None:
            kl = -0.5 * (1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        else:
            term1 = (self.mu - p.mu) / (p.sigma + 1e-8)
            term2 = self.sigma / (p.sigma + 1e-8)
            kl = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
        return kl

    def mode(self):
        return self.mu
    
class EndpointDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_dim = args.hidden_dim
        self.future_length = 1# for endpoint prediction only
        # self.MLP = MLP(input_dim = 128, output_dim = 2, hidden_size=32)
        self.MLP = MLP(96, 2, hidden_dims=(512, 256))


    def forward(self, past_feature, z, batch_size_curr, agent_num_perscene, cur_location, sample_num, mode='train'):
        agent_num = batch_size_curr * agent_num_perscene
        past_feature = past_feature.view(-1,sample_num,past_feature.shape[-1])

        z_in = z.view(-1, sample_num, z.shape[-1])

        hidden = torch.cat((past_feature,z_in),dim=-1)
        hidden = hidden.view(agent_num*sample_num,-1)
        # x_true = past_traj_repeat.clone() #torch.transpose(pre_motion_scene_norm, 0, 1)

        # prediction = torch.zeros((agent_num*sample_num, self.future_length, 2)).cuda()
        # prediction = torch.zeros((batch_size, 1, 2)).cuda()
        norm_seq = self.MLP(hidden).contiguous().view(-1, self.future_length, 2)

        cur_location_repeat = cur_location.repeat_interleave(sample_num, dim=0)
        out_seq = norm_seq + cur_location_repeat # (agent_num*sample_num,self.past_length,2)
        if mode == 'inference':
            out_seq = out_seq.view(-1,sample_num,*out_seq.shape[1:]) # (agent_num,sample_num,self.past_length,2)
        return out_seq
class DecomposeBlock(nn.Module):
    '''
    Balance between reconstruction task and prediction task.
    '''
    def __init__(self, past_len, future_len, input_dim):
        super(DecomposeBlock, self).__init__()
        # * HYPER PARAMETERS
        channel_in = 2
        channel_out = 32
        dim_kernel = 3
        dim_embedding_key = 96
        self.past_len = past_len
        self.future_len = future_len

        self.conv_past = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.encoder_past = nn.GRU(channel_out, dim_embedding_key, 1, batch_first=True)
        
        self.decoder_y = MLP(dim_embedding_key + input_dim, future_len * 2, hidden_dims=(512, 256))
        self.decoder_x = MLP(dim_embedding_key + input_dim, past_len * 2, hidden_dims=(512, 256))

        self.relu = nn.ReLU()

        # kaiming initialization
        self.init_parameters()

    def init_parameters(self):
        nn.init.kaiming_normal_(self.conv_past.weight)
        nn.init.kaiming_normal_(self.encoder_past.weight_ih_l0)
        nn.init.kaiming_normal_(self.encoder_past.weight_hh_l0)

        nn.init.zeros_(self.conv_past.bias)
        nn.init.zeros_(self.encoder_past.bias_ih_l0)
        nn.init.zeros_(self.encoder_past.bias_hh_l0)


    def forward(self, x_true, x_hat, f):
        '''
        >>> Input:
            x_true: N, T_p, 2
            x_hat: N, T_p, 2
            f: N, D

        >>> Output:
            x_hat_after: N, T_p, 2
            y_hat: n, T_f, 2
        '''
        x_ = x_true - x_hat
        x_ = torch.transpose(x_, 1, 2)
        
        past_embed = self.relu(self.conv_past(x_))
        past_embed = torch.transpose(past_embed, 1, 2)

        _, state_past = self.encoder_past(past_embed)
        state_past = state_past.squeeze(0)

        input_feat = torch.cat((f, state_past), dim=1)

        x_hat_after = self.decoder_x(input_feat).contiguous().view(-1, self.past_len, 2)
        y_hat = self.decoder_y(input_feat).contiguous().view(-1, self.future_len, 2)
        
        return x_hat_after, y_hat

class DecoderFinal(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model_dim = args.hidden_dim
        self.decode_way = 'RES'
        scale_num = 2 + len(self.args.hyper_scales)
        
        self.num_decompose = 2
        # input_dim = scale_num*self.model_dim +self.args.zdim
        self.past_length = self.args.past_length
        self.future_length = self.args.future_length

        self.decompose = nn.ModuleList([DecomposeBlock(self.args.past_length, self.args.future_length, 416) for _ in range(self.num_decompose)])

    def forward(self, past_feature, z, batch_size_curr, agent_num_perscene, past_traj, cur_location, sample_num, mode='train'):
        agent_num = batch_size_curr * agent_num_perscene
        past_traj_repeat = past_traj.repeat_interleave(sample_num, dim=0)
        past_feature = past_feature.view(-1,sample_num,past_feature.shape[-1])

        z_in = z.view(-1, sample_num, z.shape[-1])

        hidden = torch.cat((past_feature,z_in),dim=-1)
        hidden = hidden.view(agent_num*sample_num,-1)
        x_true = past_traj_repeat.clone() #torch.transpose(pre_motion_scene_norm, 0, 1)

        x_hat = torch.zeros_like(x_true)
        batch_size = x_true.size(0)
        prediction = torch.zeros((batch_size, self.future_length, 2)).cuda()
        reconstruction = torch.zeros((batch_size, self.past_length, 2)).cuda()

        for i in range(self.num_decompose):
            x_hat, y_hat = self.decompose[i](x_true, x_hat, hidden)
            prediction += y_hat
            reconstruction += x_hat
        norm_seq = prediction.view(agent_num*sample_num,self.future_length,2)
        recover_pre_seq = reconstruction.view(agent_num*sample_num,self.past_length,2)

        # norm_seq = norm_seq.permute(2,0,1,3).view(self.future_length, agent_num * sample_num,2)

        cur_location_repeat = cur_location.repeat_interleave(sample_num, dim=0)
        out_seq = norm_seq + cur_location_repeat # (agent_num*sample_num,self.past_length,2)
        if mode == 'inference':
            out_seq = out_seq.view(-1,sample_num,*out_seq.shape[1:]) # (agent_num,sample_num,self.past_length,2)
        return out_seq,recover_pre_seq          

class CVAEMart(nn.Module):
    def __init__(self, args):
        super(CVAEMart, self).__init__()
        self.args = args

        self.ztype = 'gaussian'
        self.zdim = 32

        self.past_traj_encoder = MARTModule(args).cuda()

        self.future_traj_encoder = MARTModule(args, future=True).cuda()

        self.coarse_traj_encoder = MARTModule(args, future=True).cuda()

        self.coarse_traj_encoder_mlp = CoarseEncoderMLP(args, future=True).cuda()

        # self.decoder_final = DecoderFinal(args).cuda()

        self.endpoint_decoder = EndpointDecoder(args).cuda()

        self.decoder_final =  DecoderFinal(args).cuda()

        module_args = {
            'num_layers': 1,
            'num_heads': args.num_heads,
            'node_dim': args.model_dim,
            'node_hidden_dim': args.hidden_dim,
            'edge_dim': args.model_dim,
            'edge_hidden_dim_1': args.hidden_dim,
            'edge_hidden_dim_2': args.hidden_dim,
            'dropout': args.dropout,
        }

        self.out_mlp = MLP(
            128,
            128,
            hidden_dims=(
                args.hidden_dim,
                args.hidden_dim,
            )
        )
        self.qz_layer = nn.Linear(self.out_mlp.out_dim, 2 * self.zdim)

        initialize_weights(self.qz_layer.modules())

        self.out_mlp_social = MLP(
            480,
            128,
            hidden_dims=(
                args.hidden_dim,
                args.hidden_dim
            )
        )
        self.qz_layer_social = nn.Linear(self.out_mlp_social.out_dim, 2 * self.zdim)
        initialize_weights(self.qz_layer_social.modules())

    def forward(self, x_abs, x_rel, y_abs, y_rel):

        batch_size, agent_num, length, _ = x_abs.shape
        cur_location = x_abs[:, :, [-1], :].contiguous().view(batch_size*agent_num,1, -1).contiguous()

        past_n_final, past_n_initial, past_n_pair, past_n_group = self.past_traj_encoder(x_abs, x_rel)
        future_n_final, future_n_initial, future_n_pair, future_n_group = self.future_traj_encoder(y_abs, y_rel, offset=self.args.past_length )
        

        

        past_social = torch.cat(( past_n_pair, past_n_group), dim=-1)
        future_social = torch.cat(( future_n_pair, future_n_group), dim=-1)

        h = torch.cat((past_n_initial, future_n_initial), dim=-1)

        out = self.out_mlp(h)
        qz_param_int = self.qz_layer(out)

        past_n_final = past_n_final.view(batch_size*agent_num, -1)
        
         ### q dist ###
        if self.ztype == 'gaussian':
            qz_distribution = Normal(params=qz_param_int)
        else:
            ValueError('Unknown hidden distribution!')
        qz_sampled = qz_distribution.rsample()

        ### p dist ###
        if self.ztype == 'gaussian':
            pz_distribution = Normal(mu=torch.zeros(past_n_final.shape[0], self.zdim).to(past_n_final.device), 
                                    logvar=torch.zeros(past_n_final.shape[0], self.zdim).to(past_n_final.device))
        else:
            ValueError('Unknown hidden distribution!')

        pred_endpoint = self.endpoint_decoder(past_n_initial,qz_sampled,batch_size,agent_num,cur_location,sample_num=1)

        interpolated_future_traj = interpolate_between_past_future(x_abs.view(batch_size*agent_num, -1, 2), pred_endpoint, 12)

        coarse_future_vel = interpolated_future_traj - torch.cat([x_abs.view(batch_size*agent_num, -1, 2)[:,[-1]], interpolated_future_traj[:,:-1, :]],dim=1)
        
        coarse_future_input = torch.cat((interpolated_future_traj,coarse_future_vel),dim=-1)

        coarse_n_final, coarse_n_initial, coarse_n_pair, coarse_n_group = self.coarse_traj_encoder(interpolated_future_traj.view(batch_size, agent_num, -1, 2), coarse_future_vel.view(batch_size, agent_num, -1, 2), offset=self.args.past_length)
        # coarse_n_final = self.coarse_traj_encoder_mlp(interpolated_future_traj.view(batch_size, agent_num, -1, 2), coarse_future_vel.view(batch_size, agent_num, -1, 2), offset=self.args.past_length)


        ### p dist for best 20 loss ###
        sample_num = 20

        past_int_feature_repeat = past_n_initial.repeat_interleave(sample_num, dim=0)
        # past_con_feature_repeat = int_feature.repeat_interleave(sample_num, dim=0)
        if self.ztype == 'gaussian':
            pz_distribution_diverse = Normal(mu=torch.zeros(past_int_feature_repeat.shape[0], self.zdim).to(past_n_initial.device), 
                                    logvar=torch.zeros(past_int_feature_repeat.shape[0], self.zdim).to(past_n_initial.device))
        else:
            ValueError('Unknown hidden distribution!')

        pz_sampled = pz_distribution_diverse.rsample()
        # z = pz_sampled

        diverse_pred_endpoint = self.endpoint_decoder(past_int_feature_repeat,pz_sampled,batch_size,agent_num,cur_location,sample_num=sample_num,mode='inference')
        
        diverse_pred_endpoint_reshape = diverse_pred_endpoint.reshape(batch_size*agent_num*sample_num, 1, 2)

        diverse_pred_interpolated = interpolate_between_past_future(x_abs.repeat_interleave(sample_num, dim=0).reshape(batch_size*agent_num*sample_num, -1, 2), diverse_pred_endpoint_reshape, 12)

        diverse_pred_interpolated_vel = diverse_pred_interpolated - torch.cat([x_abs.repeat_interleave(sample_num, dim=0).reshape(batch_size*agent_num*sample_num, -1, 2)[:,[-1]], diverse_pred_interpolated[:,:-1, :]],dim=1)
        
        diverse_pred_input = torch.cat((diverse_pred_interpolated,diverse_pred_interpolated_vel),dim=-1)

        diverse_n_final, diverse_n_initial, diverse_n_pair, diverse_n_group = self.coarse_traj_encoder(diverse_pred_interpolated.view(batch_size * sample_num, agent_num, -1, 2), diverse_pred_interpolated_vel.view(batch_size * sample_num, agent_num, -1, 2), offset=self.args.past_length)
        # diverse_n_final = self.coarse_traj_encoder_mlp(diverse_pred_interpolated.view(batch_size * sample_num, agent_num, -1, 2), diverse_pred_interpolated_vel.view(batch_size * sample_num, agent_num, -1, 2), offset=self.args.past_length)

        h_social = torch.cat((past_social,  future_social, qz_sampled, coarse_n_final),dim=-1)
        h_social = self.out_mlp_social(h_social)
        qz_param_social = self.qz_layer_social(h_social)

        ### q dist ###
        if self.ztype == 'gaussian':
            qz_distribution_social = Normal(params=qz_param_social)
        else:
            ValueError('Unknown hidden distribution!')
        qz_sampled_social = qz_distribution_social.rsample()

        ### p dist ###
        if self.ztype == 'gaussian':
            pz_distribution_social = Normal(mu=torch.zeros(past_social.shape[0], self.zdim).to(past_social.device), 
                                    logvar=torch.zeros(past_social.shape[0], self.zdim).to(past_social.device))
        else:
            ValueError('Unknown hidden distribution!')

        condition_feature = torch.cat((past_n_final, coarse_n_final), dim=-1)

        pred_traj_final,recover_traj_final = self.decoder_final(condition_feature,qz_sampled_social,batch_size,agent_num,x_abs.view(batch_size*agent_num, self.args.past_length, -1),cur_location,sample_num=1)
        

        ### p dist for best 20 loss ###
        sample_num = 20
        past_ped_feature_repeat = past_social.repeat_interleave(sample_num, dim=0)
        # past_con_feature_repeat = condition_feature.repeat_interleave(sample_num, dim=0)
        # past_con_feature_repeat = int_feature.repeat_interleave(sample_num, dim=0)
        if self.ztype == 'gaussian':
            pz_distribution_social_diverse = Normal(mu=torch.zeros(past_ped_feature_repeat.shape[0], self.zdim).to(past_social.device), 
                                    logvar=torch.zeros(past_ped_feature_repeat.shape[0], self.zdim).to(past_social.device))
        else:
            ValueError('Unknown hidden distribution!')

        pz_sampled_social = pz_distribution_social_diverse.rsample()
        # z = pz_sampled_social
        diverse_condition_feature = torch.cat((past_n_final.repeat_interleave(sample_num, dim=0), diverse_n_final), dim=-1)
        diverse_pred_traj_final,_ = self.decoder_final(diverse_condition_feature,pz_sampled_social,batch_size,agent_num,x_abs.view(batch_size*agent_num, self.args.past_length, -1),cur_location,sample_num=20,mode='inference')

#         loss_coarse_future_align = F.mse_loss(
#         F.normalize(coarse_n_final, dim=-1),
#         F.normalize(future_n_final.detach(), dim=-1)
# )
        return pred_endpoint, diverse_pred_endpoint, pred_traj_final, recover_traj_final, diverse_pred_traj_final, pz_distribution, qz_distribution, pz_distribution_social, qz_distribution_social
    

class PairwiseEdgeUncertainty(nn.Module):
    def __init__(self, edge_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.mc_edge_fc1 = nn.Linear(edge_dim, hidden_dim)
        self.mc_edge_fc2 = nn.Linear(hidden_dim, edge_dim)
        self.edge_alea_head = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim)
        )
        self.edge_message = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim)
        )
        self.edge_score = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.total_uncertainty_mlp = nn.Sequential(
        nn.Linear(edge_dim * 2, edge_dim),
        nn.ReLU(),
        nn.Linear(edge_dim, edge_dim),
        nn.Softplus()
        )

        self.edge_unc_fusion = nn.Sequential(
        nn.Linear(edge_dim * 5, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, edge_dim)
        )
    def fuse_uncertainty_edge(
        self,
        sampled_edge,
        mu_edge,
        U_a_edge,
        U_e_edge,
        total_var_edge
    ):
        if sampled_edge.dim() != 4:
            raise ValueError(
                "sampled_edge must have shape [B, N, N, D], "
                f"but received {tuple(sampled_edge.shape)}."
            )
        edge_unc_input = torch.cat(
            [
                sampled_edge,
                mu_edge,
                U_a_edge,
                U_e_edge,
                total_var_edge
            ],
            dim=-1
        )
        uncertainty_aware_edge = self.edge_unc_fusion(
            edge_unc_input
        )
        num_agents = sampled_edge.size(1)
        self_mask = self.create_self_mask(
            num_agents=num_agents,
            device=sampled_edge.device
        )
        uncertainty_aware_edge = uncertainty_aware_edge.masked_fill(
            self_mask,
            0.0
        )
        return uncertainty_aware_edge
    
    def stochastic_edge_forward(self, edge_feat, force_dropout=False):
        h = self.mc_edge_fc1(edge_feat)
        h = F.relu(h)
        h = F.dropout(
            h,
            p=self.dropout,
            training=(self.training or force_dropout)
        )
        h = self.mc_edge_fc2(h)
        return edge_feat + h
    def create_self_mask(self, num_agents, device):
        return torch.eye(
            num_agents,
            device=device,
            dtype=torch.bool
        ).view(1, num_agents, num_agents, 1)
    def compute_edge_distribution(self, edge_feat, mc_samples=5):
        if edge_feat.dim() != 4:
            raise ValueError(
                "edge_feat must have shape [B, N, N, D], "
                f"but received {tuple(edge_feat.shape)}."
            )
        batch_size, num_agents, num_neighbors, edge_dim = edge_feat.shape
        if num_agents != num_neighbors:
            raise ValueError(
                "The two agent dimensions of edge_feat must be equal, "
                f"but received {num_agents} and {num_neighbors}."
            )
        if edge_dim != self.edge_dim:
            raise ValueError(
                f"Expected edge dimension {self.edge_dim}, "
                f"but received {edge_dim}."
            )
        if mc_samples < 1:
            raise ValueError("mc_samples must be at least 1.")
        mc_edge_samples = []
        for _ in range(mc_samples):
            edge_m = self.stochastic_edge_forward(
                edge_feat,
                force_dropout=True
            )
            mc_edge_samples.append(edge_m)
        mc_edge_stack = torch.stack(
            mc_edge_samples,
            dim=0
        )
        mu_edge = mc_edge_stack.mean(dim=0)
        if mc_samples > 1:
            U_e_edge = mc_edge_stack.var(
                dim=0,
                unbiased=False
            )
        else:
            U_e_edge = torch.zeros_like(mu_edge)
        U_a_edge_raw = self.edge_alea_head(edge_feat)
        U_a_edge = F.softplus(U_a_edge_raw) + 1e-6
        self_mask = self.create_self_mask(
            num_agents=num_agents,
            device=edge_feat.device
        )
        mu_edge = mu_edge.masked_fill(
            self_mask,
            0.0
        )
        U_e_edge = U_e_edge.masked_fill(
            self_mask,
            0.0
        )
        U_a_edge = U_a_edge.masked_fill(
            self_mask,
            0.0
        )
        uncertainty_input = torch.cat([U_a_edge, U_e_edge],dim=-1)
        total_var_edge = self.total_uncertainty_mlp(uncertainty_input)
        # total_var_edge = U_a_edge + U_e_edge
        total_var_edge = total_var_edge.clamp(
            min=1e-6,
            max=1e3
        )
        total_var_edge = total_var_edge.masked_fill(
            self_mask,
            0.0
        )
        return {
            "mc_edge_stack": mc_edge_stack,
            "mu_edge": mu_edge,
            "U_e_edge": U_e_edge,
            "U_a_edge": U_a_edge,
            "total_var_edge": total_var_edge
        }
    def sample_edges(self, mu_edge, total_var_edge, num_edge_samples):
        if num_edge_samples < 1:
            raise ValueError("num_edge_samples must be at least 1.")
        batch_size, num_agents, _, edge_dim = mu_edge.shape
        epsilon = torch.randn(
            batch_size,
            num_agents,
            num_agents,
            num_edge_samples,
            edge_dim,
            device=mu_edge.device,
            dtype=mu_edge.dtype
        )
        sampled_edges = (
            mu_edge.unsqueeze(3)
            + torch.sqrt(total_var_edge).unsqueeze(3) * epsilon
        )
        self_mask = torch.eye(
            num_agents,
            device=mu_edge.device,
            dtype=torch.bool
        ).view(1, num_agents, num_agents, 1, 1)
        sampled_edges = sampled_edges.masked_fill(
            self_mask,
            0.0
        )
        return sampled_edges
    def aggregate_sampled_edge(self, sampled_edge):
        if sampled_edge.dim() != 4:
            raise ValueError(
                "sampled_edge must have shape [B, N, N, D], "
                f"but received {tuple(sampled_edge.shape)}."
            )
        _, num_agents, _, _ = sampled_edge.shape
        self_mask = self.create_self_mask(
            num_agents=num_agents,
            device=sampled_edge.device
        )
        edge_logits = self.edge_score(sampled_edge)
        edge_logits = edge_logits.masked_fill(
            self_mask,
            torch.finfo(edge_logits.dtype).min
        )
        edge_attn = torch.softmax(
            edge_logits,
            dim=2
        )
        edge_attn = edge_attn.masked_fill(
            self_mask,
            0.0
        )
        edge_messages = self.edge_message(sampled_edge)
        edge_messages = edge_messages.masked_fill(
            self_mask,
            0.0
        )
        node_delta = (
            edge_attn * edge_messages
        ).sum(dim=2)
        return node_delta, edge_attn, edge_messages
    def forward(self, edge_feat, mc_samples=5, num_edge_samples=20):
        distribution = self.compute_edge_distribution(
            edge_feat=edge_feat,
            mc_samples=mc_samples
        )
        sampled_edges = self.sample_edges(
            mu_edge=distribution["mu_edge"],
            total_var_edge=distribution["total_var_edge"],
            num_edge_samples=num_edge_samples
        )
        node_delta_samples = []
        edge_attn_samples = []
        edge_message_samples = []
        uncertainty_aware_edge_samples = []
        for sample_index in range(num_edge_samples):
            sampled_edge_s = sampled_edges[
                :,
                :,
                :,
                sample_index,
                :
            ]
            uncertainty_aware_edge_s = self.fuse_uncertainty_edge(
                sampled_edge=sampled_edge_s,
                mu_edge=distribution["mu_edge"],
                U_a_edge=distribution["U_a_edge"],
                U_e_edge=distribution["U_e_edge"],
                total_var_edge=distribution["total_var_edge"]
            )
            node_delta_s, edge_attn_s, edge_messages_s = self.aggregate_sampled_edge(
                uncertainty_aware_edge_s
            )
            uncertainty_aware_edge_samples.append(
                uncertainty_aware_edge_s
            )
            node_delta_samples.append(
                node_delta_s
            )
            edge_attn_samples.append(
                edge_attn_s
            )
            edge_message_samples.append(
                edge_messages_s
            )
        # for sample_index in range(num_edge_samples):
        #     sampled_edge_s = sampled_edges[
        #         :,
        #         :,
        #         :,
        #         sample_index,
        #         :
        #     ]
        #     node_delta_s, edge_attn_s, edge_messages_s = self.aggregate_sampled_edge(
        #         sampled_edge_s
        #     )
        #     node_delta_samples.append(node_delta_s)
        #     edge_attn_samples.append(edge_attn_s)
        #     edge_message_samples.append(edge_messages_s)
        node_delta_stack = torch.stack(
            node_delta_samples,
            dim=2
        )
        edge_attn_stack = torch.stack(
            edge_attn_samples,
            dim=3
        )
        edge_message_stack = torch.stack(
            edge_message_samples,
            dim=3
        )
        uncertainty_aware_edge_stack = torch.stack(
        uncertainty_aware_edge_samples,
        dim=3
    )
        aux = {
            "mc_edge_stack": distribution["mc_edge_stack"],
            "mu_edge": distribution["mu_edge"],
            "U_e_edge": distribution["U_e_edge"],
            "U_a_edge": distribution["U_a_edge"],
            "total_var_edge": distribution["total_var_edge"],
            "sampled_edges": sampled_edges,
            "node_delta_stack": node_delta_stack,
            "edge_attn_stack": edge_attn_stack,
            "edge_message_stack": edge_message_stack,
            "uncertainty_aware_edge_stack": uncertainty_aware_edge_stack
        }
        return sampled_edges, node_delta_stack, aux


class UNCERTAINMARTREASONING(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        module_args = {
            "num_layers": 1,
            "num_heads": args.num_heads,
            "node_dim": args.model_dim,
            "node_hidden_dim": args.hidden_dim,
            "edge_dim": args.model_dim,
            "edge_hidden_dim_1": args.hidden_dim,
            "edge_hidden_dim_2": args.hidden_dim,
            "dropout": args.dropout
        }

        # ==================================================
        # Input
        # ==================================================
        self.input_dim = len(args.inputs)
        self.input_fc = nn.Linear(self.input_dim, args.model_dim)
        self.input_fc2 = nn.Linear(args.model_dim * args.past_length, args.model_dim)
        self.pos_encoder = PositionalAgentEncoding(args.model_dim, 0.1, concat=True)

        # ==================================================
        # Main transformer
        # ==================================================
        self.pair_encoders = nn.ModuleList([
            RT(**module_args) if i == 0 else RTNoEdgeInit(**module_args)
            for i in range(args.num_layers)
        ])

        group_args = dict(module_args)
        group_args["function_type"] = args.function_type

        self.hyper_encoders = nn.ModuleList([
            HRT(**group_args) if i == 0 else HRTNoEdgeInit(**group_args)
            for i in range(args.num_layers)
        ])

        # ==================================================
        # Epistemic transformer
        # ==================================================
        self.epi_samples = getattr(args, "epi_samples", 20)
        self.num_epi_decoders = getattr(args, "num_epi_decoders", 20)
        self.epi_transformer_dropout = getattr(args, "epi_transformer_dropout", 0.2)

        epi_module_args = dict(module_args)
        epi_module_args["dropout"] = self.epi_transformer_dropout

        self.epi_pair_encoders = nn.ModuleList([
            RT(**epi_module_args) if i == 0 else RTNoEdgeInit(**epi_module_args)
            for i in range(args.num_layers)
        ])

        epi_group_args = dict(epi_module_args)
        epi_group_args["function_type"] = args.function_type

        self.epi_hyper_encoders = nn.ModuleList([
            HRT(**epi_group_args) if i == 0 else HRTNoEdgeInit(**epi_group_args)
            for i in range(args.num_layers)
        ])

        # ==================================================
        # Epistemic decoders
        # ==================================================
        self.epistemic_decoders = nn.ModuleList([
            EpistemicDecoder(args)
            for _ in range(self.num_epi_decoders)
        ])

        # ==================================================
        # Main decoders
        # ==================================================
        self.num_decoders = getattr(args, "num_decoders", args.sample_k)
        self.final_dim = args.model_dim * 3

        for k in range(self.num_decoders):
            self.add_module(f"head_{k}", Decoder(args))

    def build_input(self, x_abs, x_rel):
        xs = []
        if "pos_x" in self.args.inputs and "pos_y" in self.args.inputs: xs.append(x_abs)
        if "vel_x" in self.args.inputs and "vel_y" in self.args.inputs: xs.append(x_rel)
        if not xs: raise ValueError("No valid input components.")
        return torch.cat(xs, dim=-1)

    def run_pair_branch(self, n_initial, encoders):
        e = encoders[0].build_initial_edge(n_initial)
        n, e = encoders[0].reasoning(node_features=n_initial, edge_features=e)
        for i in range(1, len(encoders)):
            n, e = encoders[i].reasoning(node_features=n, edge_features=e)
        return n, e

    def run_group_branch(self, n_initial, encoders):
        e, G = encoders[0].build_initial_group(n_initial)
        n, e, G = encoders[0].reasoning(
            node_features=n_initial,
            edge_features=e,
            G=G
        )

        for i in range(1, len(encoders)):
            n, e, G = encoders[i].reasoning(
                node_features=n,
                edge_features=e,
                G=G
            )

        return n, e, G

    def set_mc_dropout(self, module, enabled=True):
        for m in module.modules():
            if isinstance(m, nn.Dropout):
                m.train(enabled)

    def run_epistemic_transformer(self, n_initial):
        # model.eval() 时也打开 epistemic transformer dropout
        self.set_mc_dropout(self.epi_pair_encoders, True)
        self.set_mc_dropout(self.epi_hyper_encoders, True)

        n_pair, e_pair = self.run_pair_branch(
            n_initial,
            self.epi_pair_encoders
        )

        n_group, e_group, G = self.run_group_branch(
            n_initial,
            self.epi_hyper_encoders
        )

        epi_feature = torch.cat(
            [n_initial, n_pair, n_group],
            dim=-1
        )

        return epi_feature, n_pair, e_pair, n_group, e_group

    def run_epistemic_branch(self, n_initial, cur_pos_bn):
        all_samples, feature_samples = [], []

        for _ in range(self.epi_samples):
            epi_feature, _, _, _, _ = self.run_epistemic_transformer(
                n_initial
            )

            feature_samples.append(epi_feature)

            head_outputs = [
                decoder(
                    epi_feature,
                    cur_pos=cur_pos_bn,
                    force_dropout=True
                )
                for decoder in self.epistemic_decoders
            ]

            all_samples.append(
                torch.stack(head_outputs, dim=2)
            )
            # [B,N,E,T,2]

        epi_samples = torch.stack(all_samples, dim=3)
        # [B,N,E,M,T,2]

        epi_feature_samples = torch.stack(
            feature_samples,
            dim=2
        )
        # [B,N,M,3D]

        return epi_samples, epi_feature_samples

    def run_all_decoders(
        self,
        feature,
        uncertainty,
        cur_pos,
        B,
        N
    ):
        feature = feature.reshape(B * N, -1)
        uncertainty = uncertainty.reshape(B * N, 1)

        outputs = []

        for k in range(self.num_decoders):
            pred = self._modules[f"head_{k}"](
                feature,
                cur_pos,
                uncertainty
            )

            outputs.append(
                pred.view(
                    B,
                    N,
                    self.args.future_length,
                    2
                )
            )

        return torch.stack(outputs, dim=2)

    def forward(
        self,
        x_abs,
        x_rel,
        target=None,
        return_aux=False
    ):
        if target is None:
            raise ValueError(
                "target is required because GT is used to "
                "select the best epistemic decoder."
            )

        if target.dim() == 5 and target.size(2) == 1:
            target = target.squeeze(2)

        B, N, T_obs, _ = x_abs.shape
        T = self.args.future_length
        M = self.epi_samples

        cur_pos_bn = x_abs[:, :, -1:, :2]
        cur_pos = cur_pos_bn.reshape(
            B * N,
            1,
            2
        ).contiguous()

        # ==================================================
        # Input encoding
        # ==================================================
        x = self.build_input(
            x_abs,
            x_rel
        ).reshape(
            B * N,
            T_obs,
            -1
        )

        x = self.input_fc(x)
        x = self.pos_encoder(
            x,
            num_a=B * N
        )

        x = x.reshape(
            B,
            N,
            T_obs,
            self.args.model_dim
        )

        n_initial = self.input_fc2(
            x.reshape(
                B,
                N,
                T_obs * self.args.model_dim
            )
        )

        # ==================================================
        # Main transformer
        # ==================================================
        n_pair, e_pair = self.run_pair_branch(
            n_initial,
            self.pair_encoders
        )

        n_group, e_group, G = self.run_group_branch(
            n_initial,
            self.hyper_encoders
        )

        n_final = torch.cat(
            [n_initial, n_pair, n_group],
            dim=-1
        )

        # ==================================================
        # Epistemic branch
        # [B,N,E,M,T,2]
        # ==================================================
        epi_samples, epi_feature_samples = self.run_epistemic_branch(
            n_initial,
            cur_pos_bn
        )

        # ==================================================
        # Convert epistemic trajectories to absolute
        # ==================================================
        if self.args.pred_rel:
            epi_abs = (
                torch.cumsum(
                    epi_samples,
                    dim=4
                )
                + cur_pos_bn.unsqueeze(2).unsqueeze(3)
            )
        else:
            epi_abs = epi_samples

        # ==================================================
        # GT selects best epistemic decoder
        # ==================================================

        # MC mean for every epistemic decoder
        epi_mean = epi_abs.mean(dim=3)
        # [B,N,E,T,2]

        epi_head_dist = torch.norm(
            epi_mean - target.unsqueeze(2),
            dim=-1
        )
        # [B,N,E,T]

        epi_head_ade = epi_head_dist.mean(dim=-1)
        # [B,N,E]

        best_epi_ade, best_epi_idx = epi_head_ade.min(
            dim=2
        )
        # [B,N]

        # ==================================================
        # Selected MC samples
        # ==================================================
        idx_samples = best_epi_idx[
            :, :, None, None, None, None
        ].expand(
            B,
            N,
            1,
            M,
            T,
            2
        )

        best_epi_samples = torch.gather(
            epi_abs,
            dim=2,
            index=idx_samples
        ).squeeze(2)
        # [B,N,M,T,2]

        # ==================================================
        # Selected MC-mean trajectory
        # ==================================================
        idx_mean = best_epi_idx[
            :, :, None, None, None
        ].expand(
            B,
            N,
            1,
            T,
            2
        )

        best_epi_mean = torch.gather(
            epi_mean,
            dim=2,
            index=idx_mean
        ).squeeze(2)
        # [B,N,T,2]

        # ==================================================
        # MC variance of all heads
        # ==================================================
        epi_var_heads = epi_abs.var(
            dim=3,
            unbiased=False
        )
        # [B,N,E,T,2]

        epi_unc_t_heads = torch.sqrt(
            epi_var_heads.sum(
                dim=-1
            ).clamp_min(1e-8)
        )
        # [B,N,E,T]

        epi_unc_heads = epi_unc_t_heads.mean(
            dim=-1
        )
        # [B,N,E]

        epi_unc_final_heads = epi_unc_t_heads[
            :, :, :, -1
        ]
        # [B,N,E]

        # ==================================================
        # GT-best head uncertainty
        # ==================================================
        idx_t = best_epi_idx[
            :, :, None, None
        ].expand(
            B,
            N,
            1,
            T
        )

        epi_unc_t = torch.gather(
            epi_unc_t_heads,
            dim=2,
            index=idx_t
        ).squeeze(2)
        # [B,N,T]

        idx_var = best_epi_idx[
            :, :, None, None, None
        ].expand(
            B,
            N,
            1,
            T,
            2
        )

        epi_var = torch.gather(
            epi_var_heads,
            dim=2,
            index=idx_var
        ).squeeze(2)
        # [B,N,T,2]

        epi_unc = epi_unc_t.mean(
            dim=-1,
            keepdim=True
        )
        # [B,N,1]

        epi_unc_final = epi_unc_t[
            :, :, -1:
        ]
        # [B,N,1]

        # ==================================================
        # Normalize
        # ==================================================
        unc_scale = epi_unc.mean().detach().clamp_min(
            1e-6
        )

        epi_unc_norm = torch.log1p(
            epi_unc / unc_scale
        )

        # main prediction loss does not directly optimize U
        uncertainty_for_main = epi_unc_norm.detach()

        # ==================================================
        # Main decoders
        # ==================================================
        out = self.run_all_decoders(
            feature=n_final,
            uncertainty=uncertainty_for_main,
            cur_pos=cur_pos,
            B=B,
            N=N
        )
        # [B,N,K,T,2]

        out = out.unsqueeze(2)
        # [B,N,1,K,T,2]

        aux = {
            "epi_samples": epi_samples,
            "epi_abs": epi_abs,
            "epi_mean": epi_mean,

            "epi_feature_samples": epi_feature_samples,

            "epi_head_ade": epi_head_ade,
            "best_epi_idx": best_epi_idx,
            "best_epi_ade": best_epi_ade,

            "best_epi_samples": best_epi_samples,
            "best_epi_mean": best_epi_mean,

            "epi_var_heads": epi_var_heads,
            "epi_unc_t_heads": epi_unc_t_heads,
            "epi_unc_heads": epi_unc_heads,
            "epi_unc_final_heads": epi_unc_final_heads,

            "epi_var": epi_var,
            "epi_unc_t": epi_unc_t,
            "epi_unc": epi_unc,
            "epi_unc_final": epi_unc_final,
            "epi_unc_norm": epi_unc_norm,

            "n_initial": n_initial,
            "n_pair": n_pair,
            "e_pair": e_pair,
            "n_group": n_group,
            "e_group": e_group,
            "n_final": n_final
        }

        return (
            (out, aux)
            if return_aux
            else out
        )

class InteractionUncertainty(nn.Module):
    def __init__(self, edge_dim, hidden_dim, dropout=0.1, sample_scale=1.0):
        super().__init__()
        self.dropout = dropout
        self.sample_scale = sample_scale
        self.mc_edge_fc1 = nn.Linear(edge_dim, hidden_dim)
        self.mc_edge_fc2 = nn.Linear(hidden_dim, edge_dim)

    def stochastic_edge_forward(self, edge_feat, force_dropout=False):
        h = F.relu(self.mc_edge_fc1(edge_feat))
        h = F.dropout(
            h,
            p=self.dropout,
            training=(self.training or force_dropout)
        )
        return 0.1 * torch.tanh(self.mc_edge_fc2(h))

    def compute_edge_distribution(self, edge_feat, mc_samples=20, mask=None):
        mc_residual = torch.stack([
            self.stochastic_edge_forward(edge_feat, force_dropout=True)
            for _ in range(mc_samples)
        ], dim=0)  # [M,B,N,N,D]

        mc_residual = mc_residual - mc_residual.mean(dim=0, keepdim=True)

        mu_edge = edge_feat

        U_e_edge = (
            mc_residual.var(dim=0, unbiased=False)
            if mc_samples > 1
            else torch.zeros_like(edge_feat)
        )

        if mask is not None:
            mu_edge = mu_edge.masked_fill(mask, 0.0)
            U_e_edge = U_e_edge.masked_fill(mask, 0.0)
            mc_residual = mc_residual.masked_fill(
                mask.unsqueeze(0), 0.0
            )

        return {
            "mu_edge": mu_edge,
            "U_e_edge": U_e_edge,
            "mc_residual": mc_residual
        }

    def sample_edges(self, mu_edge, mc_residual, num_edge_samples=20, mask=None):
        M = mc_residual.size(0)
        S = num_edge_samples

        coeff = torch.randn(
            S, M,
            device=mu_edge.device,
            dtype=mu_edge.dtype
        ) / math.sqrt(M)

        sampled_residual = torch.einsum(
            'sm,mbijd->sbijd',
            coeff,
            mc_residual
        )

        sampled_edges = (
            mu_edge.unsqueeze(0)
            + self.sample_scale * sampled_residual
        ).permute(1, 2, 3, 0, 4)  # [B,N,N,S,D]

        if mask is not None:
            sampled_edges = sampled_edges.masked_fill(
                mask.unsqueeze(-2), 0.0
            )

        return sampled_edges

    def forward(
        self,
        edge_feat,
        mc_samples=20,
        num_edge_samples=20,
        mask=None
    ):
        dist = self.compute_edge_distribution(
            edge_feat, mc_samples, mask
        )

        sampled_edges = self.sample_edges(
            dist["mu_edge"],
            dist["mc_residual"],
            num_edge_samples,
            mask
        )

        dist["sampled_edges"] = sampled_edges
        return sampled_edges, dist

class EpistemicDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        H = getattr(args, "unc_hidden_dim", getattr(args, "decoder_hidden_dim", args.hidden_dim))
        self.dropout = getattr(args, "unc_dropout", 0.2)
        self.fc1 = nn.Linear(args.model_dim * 3, H)
        self.fc2 = nn.Linear(H, H)
        self.out = nn.Linear(H, args.future_length * 2)

    def forward(self, x, cur_pos=None, force_dropout=True):
        h = F.relu(self.fc1(x))
        h = F.dropout(h, p=self.dropout, training=self.training or force_dropout)
        h = F.relu(self.fc2(h))
        h = F.dropout(h, p=self.dropout, training=self.training or force_dropout)
        out = self.out(h).view(x.size(0), x.size(1), self.args.future_length, 2)
        if not self.args.pred_rel and cur_pos is not None: out = out + cur_pos
        return out