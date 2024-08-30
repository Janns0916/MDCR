import json
import math
import os
from src.models.multihead_attention import MultiheadAttention
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import RGCNConv


class KGPrompt(nn.Module):
    def __init__(
        self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
        n_entity, num_relations, num_bases, edge_index, edge_type,
        n_prefix_rec=None, n_prefix_conv=None
    ):
        super(KGPrompt, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv

        entity_hidden_size = hidden_size // 2
        self.kg_encoder = RGCNConv(entity_hidden_size, entity_hidden_size, num_relations=num_relations,
                                   num_bases=num_bases)

        self.concept_edge_sets = self.concept_edge_list4GCN()
        self.concept_embeddings = self._create_entity_embeddings(29308 + 1, entity_hidden_size, 0)
        self.concept_GCN = GCNConv(entity_hidden_size, entity_hidden_size)


        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_type = nn.Parameter(edge_type, requires_grad=False)
        self.entity_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)

        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)

        self.word_proj1 = nn.Sequential(
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )
        self.word_proj2 = nn.Linear(entity_hidden_size, hidden_size)
     

        self.cross_attn1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.cross_attn2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, n_layer * n_block * hidden_size)


        self.image_proj = nn.Linear(1024, hidden_size)
        self.mha = MultiheadAttention(embed_dim=hidden_size,
                                      num_heads=8,
                                      attn_dropout=0.1)

        if self.n_prefix_rec is not None:
            self.rec_prefix_embeds = nn.Parameter(torch.empty(n_prefix_rec, hidden_size))
            nn.init.normal_(self.rec_prefix_embeds)
            self.rec_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        if self.n_prefix_conv is not None:
            self.conv_prefix_embeds = nn.Parameter(torch.empty(n_prefix_conv, hidden_size))
            nn.init.normal_(self.conv_prefix_embeds)
            self.conv_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )

    def set_and_fix_node_embed(self, node_embeds: torch.Tensor):
        self.node_embeds.data = node_embeds
        self.node_embeds.requires_grad_(False)

    def concept_edge_list4GCN(self):
        node2index = json.load(open('./data/conceptnet/key2index_3rd.json', encoding='utf-8'))
        f = open('./data/conceptnet/conceptnet_edges2nd.txt', encoding='utf-8')
        edges = set()
        stopwords = set([word.strip() for word in open('./data/conceptnet/stopwords.txt', encoding='utf-8')])
        for line in f:
            lines = line.strip().split('\t')
            entity0 = node2index[lines[1].split('/')[0]]
            entity1 = node2index[lines[2].split('/')[0]]
            if lines[1].split('/')[0] in stopwords or lines[2].split('/')[0] in stopwords:
                continue
            edges.add((entity0, entity1))
            edges.add((entity1, entity0))
        edge_set = [[co[0] for co in list(edges)], [co[1] for co in list(edges)]]
        return torch.LongTensor(edge_set).cuda()

    def _create_entity_embeddings(self, node_num, embedding_size, padding_idx):
        """Create and initialize word embeddings."""
        e = nn.Embedding(node_num, embedding_size)
        nn.init.normal_(e.weight, mean=0, std=embedding_size ** -0.5)
        nn.init.constant_(e.weight[padding_idx], 0)
        return e

    def get_word_embeds(self):
        # conceptnet 词嵌入
        node_embeds = self.concept_embeddings
        word_embeds = self.concept_GCN(node_embeds.weight, self.concept_edge_sets)
        word_embeds = self.word_proj1(word_embeds) + word_embeds
        word_embeds = self.word_proj2(word_embeds)
        return word_embeds
 -

    def get_entity_embeds(self):
        node_embeds = self.node_embeds
        entity_embeds = self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        entity_embeds = self.entity_proj2(entity_embeds)
        return entity_embeds

    def forward(self, entity_ids=None, word_ids=None, token_embeds=None, image_embeds=None, output_entity=False,
                use_rec_prefix=False, use_conv_prefix=False):
        batch_size, entity_embeds, word_embeds, entity_len, word_len, token_len = None, None, None, None, None, None

        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            entity_embeds = self.get_entity_embeds()
            entity_embeds = entity_embeds[entity_ids]  # (batch_size, entity_len, hidden_size)

        if word_ids is not None:
            batch_size, word_len = word_ids.shape[:2]
            word_embeds = self.get_word_embeds()
            word_embeds = word_embeds[word_ids]  # (batch_size, entity_len, hidden_size)


        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = self.token_proj1(token_embeds) + token_embeds  # (batch_size, token_len, hidden_size)
            token_embeds = self.token_proj2(token_embeds)

        if word_embeds is not None:
            attn_weights2 = self.cross_attn2(token_embeds) @ word_embeds.permute(0, 2, 1)
            attn_weights2 /= self.hidden_size
      

        if entity_embeds is not None and token_embeds is not None:
            attn_weights1 = self.cross_attn1(token_embeds) @ entity_embeds.permute(0, 2, 1)  # (batch_size, token_len, entity_len)
            attn_weights1 /= self.hidden_size

            if output_entity:
                token_weights = F.softmax(attn_weights1, dim=1).permute(0, 2, 1)
                prompt_embeds = token_weights @ token_embeds + entity_embeds
                prompt_len = entity_len

            else:
                entity_weights = F.softmax(attn_weights1, dim=2)
                word_weights = F.softmax(attn_weights2, dim=2)
                prompt_embeds = entity_weights @ entity_embeds + word_weights @ word_embeds + token_embeds
                prompt_len = token_len

                # entity_weights = F.softmax(attn_weights1, dim=2)
                # prompt_embeds = entity_weights @ entity_embeds + token_embeds
                # prompt_len = token_len

        elif entity_embeds is not None:
     
            prompt_embeds = entity_embeds
            prompt_len = entity_len

        else:
    
            prompt_embeds = token_embeds
            prompt_len = token_len

        if self.n_prefix_rec is not None and use_rec_prefix:

            if image_embeds is not None:  
               
                image_embeds = self.image_proj(image_embeds)  # 64*768
              
                image_embeds = image_embeds.unsqueeze(1)
                image_embeds = image_embeds.repeat(1, prompt_embeds.shape[1], 1)  
                prompt_embeds, _ = self.mha(query=prompt_embeds, key=image_embeds, value=image_embeds)

         
            prefix_embeds = self.rec_prefix_proj(self.rec_prefix_embeds) + self.rec_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
  
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_rec
        if self.n_prefix_conv is not None and use_conv_prefix:
    
            prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
      
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_conv


        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds

        prompt_embeds = self.prompt_proj2(prompt_embeds)
        prompt_embeds = prompt_embeds.reshape(
            batch_size, prompt_len, self.n_layer, self.n_block, self.n_head, self.head_dim
        ).permute(2, 3, 0, 4, 1, 5)  # (n_layer, n_block, batch_size, n_head, prompt_len, head_dim)

        return prompt_embeds

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(load_path, map_location=torch.device('cpu')), strict=False
        )
        print(missing_keys, unexpected_keys)
