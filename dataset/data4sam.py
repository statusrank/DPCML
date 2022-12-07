import torch.utils.data as data 
from sampler import SamplerFactory

class SampleDataset(data.Dataset):

    def __init__(self, 
                 user_item_interactions,
                 num_negatives,
                 sample_method='uniform',
                 random_seed=1234):
        
        super(SampleDataset, self).__init__()

        """
        This class is adopted to generate (user, PosItem, NegItem) for training 
            according to the given sample method.
        """

        self.user_item_interactions = user_item_interactions
        self.sampler = SamplerFactory.generate_sampler(sample_method,
                                                       user_item_interactions,
                                                       num_negatives,
                                                       random_seed)
        self.user_pos_neg = self.sampler.sampling()
    def __len__(self):
        return len(self.user_pos_neg) 
    
    def __getitem__(self, idx):
        user = self.user_pos_neg[idx][0]
        pos_id = self.user_pos_neg[idx][1] 
        neg_id = self.user_pos_neg[idx][2]

        return (user, pos_id, neg_id)

    def generate_triplets_by_sampling(self):
        self.user_pos_neg = self.sampler.sampling()

    

