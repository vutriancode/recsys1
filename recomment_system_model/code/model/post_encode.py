import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Device

from Attention import Attention

class PostEncode(nn.Module):

    def __init__(self, u2e,r2e, p2e, embed_dim, contents_embed_dim, pu_history, pr_history, device="cpu"):
        super(PostEncode,self).__init__()
        self.u2e = u2e
        self.r2e = r2e
        self.p2e = p2e
        self.device = device 
        #self.contents_embedding = contents_embedding
        self.w_e = nn.Linear(contents_embed_dim, embed_dim).to(device)
        self.embed_dim = embed_dim
        self.w_1 = nn.Linear(2*embed_dim, embed_dim).to(device)
        self.w_2 = nn.Linear(embed_dim,embed_dim).to(device)
        self.attention =Attention(embed_dim,device=device)
        self.o_w = nn.Linear(2 * self.embed_dim, self.embed_dim).to(device)
        self.pu_history = pu_history 
        self.pr_history = pr_history
        #self.pr_content = pr_content


    def forward(self, nodes):

        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        for index, i in enumerate(nodes):
            i=int(i.numpy())
            j = self.pu_history[i]
            k = self.pr_history[i]
            
            #with torch.no_grad():
                #post_rep = self.contents_embedding(self.pr_content[i])
            post_rep = self.p2e[i]
            post_rep = torch.FloatTensor(post_rep, device=self.device)
            post_rep = F.relu(self.w_e(post_rep))
            u_embed, r_embed = self.u2e.weight[j], self.r2e.weight[k]
            #print(post_rep)
            number_u = len(j)
            x = torch.cat((u_embed,r_embed),1)
            x = F.relu(self.w_1(x))
            o = F.relu(self.w_2(x))
            #print(x)
            #print(o)
            att_w = self.attention(o,post_rep, number_u)

            att_history = torch.mm(o.t(), att_w)
            #print(att_history.t().reshape_as(post_rep))
            att_history = torch.cat((att_history.t().reshape_as(post_rep), post_rep))
            att_history = F.relu(self.o_w(att_history))
            embed_matrix[index] = att_history
        to_feats = embed_matrix

        return to_feats
