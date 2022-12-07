import numpy as np
from .sampler import Sampler

class PopularSampler(Sampler):
    """
    Sampler based on popularity.
    The negative examples are chosen from all items
    """

    # def _negative_sampling(self, user_ids, pos_ids, neg_ids):
    #     neg_samples = np.random.choice(neg_ids,
    #                                    size=(len(pos_ids), self.n_negatives),
    #                                    replace=False,
    #                                    p=self.item_popularities)
    #     for i, uid, negatives in zip(range(len(user_ids)),
    #                                  user_ids, neg_samples):
    #         for j, neg in enumerate(negatives):
    #             while neg in self.user_items[uid]:
    #                 neg_samples[i, j] = neg = np.random.choice(
    #                     neg_ids, p=self.item_popularities)
    #     return neg_samples
    
    def _negative_sampling(self, user_item_pairs):

        sampling_triplets = []
        candidate_neg_ids  = self._candidate_neg_ids()

        for user_id, pos in user_item_pairs:
            negs = np.random.choice(candidate_neg_ids, size=self.n_negatives, p=self.item_popularities)
            for neg in negs:
                while neg in self.user_items[user_id]:
                    neg = np.random.choice(candidate_neg_ids, p=self.item_popularities)
                sampling_triplets.append((user_id, pos, neg))
        return sampling_triplets