import random
import collections
from torch.utils.data import sampler


class NaiveIdentitySampler(sampler.Sampler):
    def __init__(self, data_source, batch_id, batch_image):
        super(NaiveIdentitySampler, self).__init__(data_source)

        self.data_source = data_source
        self.batch_image = batch_image
        self.batch_id = batch_id

        self._id2index = collections.defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self._id2index[pid].append(index)
        self.pids = list(self._id2index.keys())

    def __iter__(self):
        unique_ids = sorted(set(self.pids))
        random.shuffle(unique_ids)

        imgs = []
        for _id in unique_ids:
            imgs.extend(self._sample(self._id2index[_id], self.batch_image))
        return iter(imgs)

    def __len__(self):
        return len(self._id2index) * self.batch_image

    @staticmethod
    def _sample(population, k):
        if len(population) < k:
            population = population * k
        return random.sample(population, k)
