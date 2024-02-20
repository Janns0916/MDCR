import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
import torch
from loguru import logger


class DBpedia:
    def __init__(self, kg, debug=False):
        self.debug = debug

        # 加载文件
        self.kg_dir = os.path.join('data', kg)
        # dbpedia_subkg.json
        with open(os.path.join(self.kg_dir, 'lmkg.json'), 'r', encoding='utf-8') as f:
            self.entity_kg = json.load(f)
        # entity2id.json
        with open(os.path.join(self.kg_dir, 'lmkg_entity2id.json'), 'r', encoding='utf-8') as f:
            self.entity2id = json.load(f)
        # relation2id.json
        with open(os.path.join(self.kg_dir, 'lmkg_relation2id.json'), 'r', encoding='utf-8') as f:
            self.relation2id = json.load(f)
        # 所有项目（电影）的实体id列表
        with open(os.path.join(self.kg_dir, 'item_ids.json'), 'r', encoding='utf-8') as f:
            self.item_ids = json.load(f)

        # 所有项目（电影）的实体id列表
        with open(os.path.join(self.kg_dir, 'movie_ids.json'), 'r', encoding='utf-8') as f:
            self.movie_ids = json.load(f)

        self._process_entity_kg()

    def _process_entity_kg(self):
        edge_list = set()  # [(entity, entity, relation)]
        for entity in self.entity2id.values():
            if str(entity) not in self.entity_kg:
                continue
            for relation_and_tail in self.entity_kg[str(entity)]:
                edge_list.add((entity, relation_and_tail[1], relation_and_tail[0]))
                edge_list.add((relation_and_tail[1], entity, relation_and_tail[0]))
        edge_list = list(edge_list)
        # 上面的操作 有向变成了无向 头尾，尾头

        edge = torch.as_tensor(edge_list, dtype=torch.long)
        self.edge_index = edge[:, :2].t()  # 实体1，实体2，两者的关系
        self.edge_type = edge[:, 2]  # 所有的关系
        self.num_relations = len(self.relation2id)  # 关系有几种
        self.pad_entity_id = max(self.entity2id.values()) + 1  # 实体的pad = 最大实体ID + 1
        self.num_entities = max(self.entity2id.values()) + 2  # 实体的种类 = 最大实体ID + 2（一个是pad，另一个不知道）

        if self.debug:
            logger.debug(
                f'#edge: {len(edge)}, #relation: {self.num_relations}, '
                f'#entity: {self.num_entities}, #item: {len(self.item_ids)}, #movie: {len(self.movie_ids)}'
            )

    def get_entity_kg_info(self):
        kg_info = {
            'edge_index': self.edge_index,
            'edge_type': self.edge_type,
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'pad_entity_id': self.pad_entity_id,
            'item_ids': self.item_ids,
            'movie_ids': self.movie_ids,
        }
        return kg_info
