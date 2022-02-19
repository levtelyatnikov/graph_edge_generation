import torch

class GumbleSigmoid(torch.nn.Module):
    def __init__(self):
        super(GumbleSigmoid, self).__init__()
        self.temperature = torch.tensor([1], dtype=torch.float32).to(torch.device('cuda:2')) #torch.nn.Parameter(torch.Tensor([5]))
        self.eps = 10e-5

    def forward(self, x, tau):
        self.tau = tau
        #edge_index, edge_weights = self.sample_edges(nodes_emb=x, tau=tau)
        return self.sample_edges(nodes_emb=x, tau=tau)

    def calculate_prob_dist(self, nodes_emb):
        #self.temperature = torch.tensor(10.01).type_as(self.tau) - self.tau
        return torch.exp(-self.temperature * torch.sum(nodes_emb - nodes_emb.unsqueeze(1) + self.eps , dim=-1) ** 2)  # temperature

    def sample_edges(self, nodes_emb, tau, hard=True):
            # probs = self.calculate_prob_dist(nodes_emb)
            # probs = torch.cat([probs.unsqueeze(-1), (1-probs).unsqueeze(-1)], dim=-1)

            # new_logits = torch.log(probs + 10e-3)
            # P = torch.nn.functional.gumbel_softmax(new_logits, tau=tau, hard=True, dim= - 1)[:,:, 0]
            #probs = probs[:, :, 0]

            # Probabilities for edges
            probs = torch.clamp(1 - self.calculate_prob_dist(nodes_emb), 0, 0.9999)
            logits = - torch.log(probs + self.eps)
            
            P = self.gumbel_sigmoid_adj(logits, tau=tau, hard=hard)
    
            # Get parent child list
            childer = torch.arange(P.shape[0]).repeat(P.shape[1]).to(nodes_emb.device)
            parents = torch.arange(P.shape[0]).view(-1,1).repeat((1, P.shape[1])).flatten().to(nodes_emb.device)
            edge_index = torch.stack([childer, parents])

            # Get weight sequence of 0, 1 for edge_list
            mask = torch.clamp((P + P.T), min=0 , max=1)  #torch.clamp((P + P.T - torch.tensor([1], dtype=torch.float32).to(torch.device('cuda:2'))), min=0 , max=1)
            edge_weights = mask.view((-1, ))
            

            
            return edge_index, edge_weights, probs, mask, P
                    
    def gumbel_sigmoid_adj(self, logits, tau, hard: bool = True): 
        gumbels = (
            -torch.empty_like(
                logits,
                memory_format=torch.legacy_contiguous_format
                ).exponential_().log() # ~Gumbel(0,1)
            )
        gumbels = (logits + gumbels) / tau  
        y_soft = gumbels.sigmoid()

        if hard:
            y_hard = (y_soft >= 0.8) * 1
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret

















