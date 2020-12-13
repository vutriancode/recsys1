import torch 
import torch.nn as nn
import torch.nn.functional as F

from Attention import Attention

class UserEncode(nn.Module):

    def __init__(self, u2e, r2e, i2e, embed_dim, up_history, ur_history, device="cpu"):
        super(UserEncode,self).__init__()
        self.u2e = u2e
        self.r2e = r2e
        self.i2e = i2e
        self.device = device 
        #self.contents_embedding = contents_embedding
        #self.w_e = nn.Linear(contents_embed_dim, embed_dim)
        self.embed_dim = embed_dim
        self.w_1 = nn.Linear(2*embed_dim, embed_dim).to(device)
        self.w_2 = nn.Linear(embed_dim,embed_dim).to(device)
        self.up_history = up_history
        self.ur_history = ur_history
        self.attention =Attention(embed_dim,device)
    def forward(self, nodes):

        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)

        for index, i in enumerate(nodes):
            i = int(i.numpy())
            j = self.up_history[i]
            k = self.ur_history[i]
            u_rep = self.u2e.weight[i].to(self.device)
            p_embed = self.i2e.weight[j].to(self.device)
            #p_embed = F.relu(self.w_e(p_embed))
            r_embed = self.r2e.weight[k].to(self.device)
            #r_embed = self.post_embedding()
            number_u = len(j)

            x = torch.cat((p_embed,r_embed),1)
            x = F.relu(self.w_1(x))
            o = F.relu(self.w_2(x)).to(self.device)

            att_w = self.attention.forward(o, u_rep, number_u).to(self.device)

            att_history = torch.mm(o.t(), att_w)

            embed_matrix[index] = att_history.t()
        a = embed_matrix
        return a


