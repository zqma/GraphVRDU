
from layoutlm.data.funsd import FunsdDataset, InputFeatures
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class FUNSD4LayoutLM:

    def __init__(self, opt=None):
        self.opt = opt


    def get_data(self, shuffle=True, split='train'):
        if split=='train':
            # the LayoutLM authors already defined a specific FunsdDataset, so we are going to use this here
            train_dataset = FunsdDataset()
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                        sampler=train_sampler,
                                        batch_size=self.opt.batch_size)
            return train_dataloader

        else:
            eval_dataset = FunsdDataset()
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset,
                                        sampler=eval_sampler,
                                        batch_size=self.opt.batch_size)
            return eval_dataloader


