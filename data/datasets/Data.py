#Code from fast-reid

import glob
import os
import re
import warnings
import logging
import copy
from tabulate import tabulate
from termcolor import colored

#Support Market1501, CUHK03, DukeMTMC Person Re-ID Datasets
__all__ = ['Market1501', 'CUHK03', 'DukeMTMC', 'MSMT17', 
           'VeRi', 'VehicleID', 'VeRiWild',
           'SmallVehicleID', 'MediumVehicleID', 'LargeVehicleID', 
           'SmallVeRiWild', 'MediumVeRiWild', 'LargeVeRiWild']

class Dataset(object):
    """An abstract class representing a Dataset.
    This is the base class for ``ImageDataset`` and ``VideoDataset``.
    Args:
        train (list): contains tuples of (img_path(s), pid, camid).
        query (list): contains tuples of (img_path(s), pid, camid).
        gallery (list): contains tuples of (img_path(s), pid, camid).
        transform: transform function.
        mode (str): 'train', 'query' or 'gallery'.
        combineall (bool): combines train, query and gallery in a
            dataset for training.
        verbose (bool): show information.
    """
    _junk_pids = []  # contains useless person IDs, e.g. background, false detections

    def __init__(self, train, query, gallery, transform=None, mode='train',
                 combineall=False, verbose=True, **kwargs):
        self.train = train
        self.query = query
        self.gallery = gallery
        self.transform = transform
        self.mode = mode
        self.combineall = combineall
        self.verbose = verbose

        self.num_train_pids = self.get_num_pids(self.train)
        self.num_train_cams = self.get_num_cams(self.train)

        if self.combineall:
            self.combine_all()

        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'query':
            self.data = self.query
        elif self.mode == 'gallery':
            self.data = self.gallery
        else:
            raise ValueError('Invalid mode. Got {}, but expected to be '
                             'one of [train | query | gallery]'.format(self.mode))

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __radd__(self, other):
        """Supports sum([dataset1, dataset2, dataset3])."""
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def parse_data(self, data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.
        Args:
            data (list): contains tuples of (img_path(s), pid, camid)
        """
        pids = set()
        cams = set()
        for info in data:
            pids.add(info[1])
            cams.add(info[2])
        return len(pids), len(cams)

    def get_num_pids(self, data):
        """Returns the number of training person identities."""
        return self.parse_data(data)[0]

    def get_num_cams(self, data):
        """Returns the number of training cameras."""
        return self.parse_data(data)[1]

    def show_summary(self):
        """Shows dataset statistics."""
        pass

    def combine_all(self):
        """Combines train, query and gallery in a dataset for training."""
        combined = copy.deepcopy(self.train)

        def _combine_data(data):
            for img_path, pid, camid in data:
                if pid in self._junk_pids:
                    continue
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
                combined.append((img_path, pid, camid))

        _combine_data(self.query)
        _combine_data(self.gallery)

        self.train = combined
        self.num_train_pids = self.get_num_pids(self.train)

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.
        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not os.path.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

class ImageDataset(Dataset):
    """A base class representing ImageDataset.
    All other image datasets should subclass it.
    ``__getitem__`` returns an image given index.
    It will return ``img``, ``pid``, ``camid`` and ``img_path``
    where ``img`` has shape (channel, height, width). As a result,
    data in each batch has shape (batch_size, channel, height, width).
    """

    def __init__(self, train, query, gallery, **kwargs):
        super(ImageDataset, self).__init__(train, query, gallery, **kwargs)

    def show_train(self):
        num_train_pids, num_train_cams = self.parse_data(self.train)

        headers = ['subset', '# ids', '# images', '# cameras']
        csv_results = [['train', num_train_pids, len(self.train), num_train_cams]]

        # tabulate it
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )
        print(f"=> Loaded {self.__class__.__name__} in csv format: \n" + colored(table, "cyan"))

    def show_test(self):
        num_query_pids, num_query_cams = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)

        headers = ['subset', '# ids', '# images', '# cameras']
        csv_results = [
            ['query', num_query_pids, len(self.query), num_query_cams],
            ['gallery', num_gallery_pids, len(self.gallery), num_gallery_cams],
        ]

        # tabulate it
        table = tabulate(
            csv_results,
            tablefmt="pipe",
            headers=headers,
            numalign="left",
        )
        print(f"=> Loaded {self.__class__.__name__} in csv format: \n" + colored(table, "cyan"))

class Market1501(ImageDataset):
    """Market1501.
    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_
    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = ''
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'
    dataset_name = "market1501"

    def __init__(self, root='datasets', market1501_500k=False, **kwargs):
        # self.root = os.path.abspath(os.path.expanduser(root))
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = os.path.join(self.data_dir, 'Market-1501-v15.09.15')
        if os.path.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"Market-1501-v15.09.15".')

        self.train_dir = os.path.join(self.data_dir, 'bounding_box_train')
        self.query_dir = os.path.join(self.data_dir, 'query')
        self.gallery_dir = os.path.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = os.path.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)
        if self.market1501_500k:
            gallery += self.process_dir(self.extra_gallery_dir, is_train=False)

        super(Market1501, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data

class DukeMTMC(ImageDataset):
    """DukeMTMC-reID.
    Reference:
        - Ristani et al. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. ECCVW 2016.
        - Zheng et al. Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro. ICCV 2017.
    URL: `<https://github.com/layumi/DukeMTMC-reID_evaluation>`_
    Dataset statistics:
        - identities: 1404 (train + query).
        - images:16522 (train) + 2228 (query) + 17661 (gallery).
        - cameras: 8.
    """
    dataset_dir = 'DukeMTMC-reID'
    dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip'
    dataset_name = "dukemtmc"

    def __init__(self, root='datasets', **kwargs):
        # self.root = os.path.abspath(os.path.expanduser(root))
        self.root = root
        self.dataset_dir = os.path.join(self.root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'bounding_box_test')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super(DukeMTMC, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data

class CUHK03(ImageDataset):
    """CUHK03.
    Reference:
        Li et al. DeepReID: Deep Filter Pairing Neural Network for Person Re-identification. CVPR 2014.
    URL: `<http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html#!>`_
    Dataset statistics:
        - identities: 1360.
        - images: 13164.
        - cameras: 6.
        - splits: 20 (classic).
    """
    dataset_dir = 'cuhk03'
    dataset_url = None
    dataset_name = "cuhk03"
    def __init__(self, root='datasets', **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super(CUHK03, self).__init__(train, query, gallery, **kwargs)
    
    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data

class MSMT17(ImageDataset):
    """MSMT17.
    Reference:
        Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.
    URL: `<http://www.pkuvmc.com/publications/msmt17.html>`_
    Dataset statistics:
        - identities: 4101.
        - images: 32621 (train) + 11659 (query) + 82161 (gallery).
        - cameras: 15.
    """
    TRAIN_DIR_KEY = 'train_dir'
    TEST_DIR_KEY = 'test_dir'
    VERSION_DICT = {
        'MSMT17_V1': {
            TRAIN_DIR_KEY: 'train',
            TEST_DIR_KEY: 'test',
        },
        'MSMT17_V2': {
            TRAIN_DIR_KEY: 'mask_train_v2',
            TEST_DIR_KEY: 'mask_test_v2',
        }
    }
    dataset_url = None
    dataset_name = 'msmt17'

    def __init__(self, root='datasets', **kwargs):
        self.dataset_dir = root

        has_main_dir = False
        for main_dir in VERSION_DICT:
            if osp.exists(osp.join(self.dataset_dir, main_dir)):
                train_dir = VERSION_DICT[main_dir][TRAIN_DIR_KEY]
                test_dir = VERSION_DICT[main_dir][TEST_DIR_KEY]
                has_main_dir = True
                break
        assert has_main_dir, 'Datset folder not found'

        self.train_dir = osp.join(self.dataset_dir, main_dir, train_dir)
        self.test_dir = osp.join(self.dataset_dir, main_dir, test_dir)
        self.list_train_path = osp.join(self.dataset_dir, main_dir, 'list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, main_dir, 'list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, main_dir, 'list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, main_dir, 'list_gallery.txt')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.test_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, self.list_train_path)
        val = self.process_dir(self.train_dir, self.list_val_path)
        query = self.process_dir(self.test_dir, self.list_query_path, is_train=False)
        gallery = self.process_dir(self.test_dir, self.list_gallery_path, is_train=False)

        um_train_pids = self.get_num_pids(train)
        query_tmp = []
        for img_path, pid, camid in query:
            query_tmp.append((img_path, pid+num_train_pids, camid))
        del query
        query = query_tmp

        gallery_temp = []
        for img_path, pid, camid in gallery:
            gallery_temp.append((img_path, pid+num_train_pids, camid))
        del gallery
        gallery = gallery_temp

        # Note: to fairly compare with published methods on the conventional ReID setting,
        #       do not add val images to the training set.
        if 'combineall' in kwargs and kwargs['combineall']:
            train += val
        super(MSMT17, self).__init__(train, query, gallery, **kwargs)


    def process_dir(self, dir_path, list_path, is_train=True):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()

        data = []

        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2]) - 1  # index starts from 0
            img_path = osp.join(dir_path, img_path)
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data


class VeRi(ImageDataset):
    """VeRi.
    Reference:
        Xinchen Liu et al. A Deep Learning based Approach for Progressive Vehicle Re-Identification. ECCV 2016.
        Xinchen Liu et al. PROVID: Progressive and Multimodal Vehicle Reidentification for Large-Scale Urban Surveillance. IEEE TMM 2018.
    URL: `<https://vehiclereid.github.io/VeRi/>`_
    Dataset statistics:
        - identities: 775.
        - images: 37778 (train) + 1678 (query) + 11579 (gallery).
    """
    dataset_dir = "veri"
    dataset_name = "veri"

    def __init__(self, root='datasets', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super(VeRi, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d]+)_c(\d\d\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 776
            assert 1 <= camid <= 20
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data


class VehicleID(ImageDataset):
    """VehicleID.
    Reference:
        Liu et al. Deep relative distance learning: Tell the difference between similar vehicles. CVPR 2016.
    URL: `<https://pkuml.org/resources/pku-vehicleid.html>`_
    Train dataset statistics:
        - identities: 13164.
        - images: 113346.
    """
    dataset_dir = "vehicleid"
    dataset_name = "vehicleid"

    def __init__(self, root='datasets', test_list='', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.image_dir = osp.join(self.dataset_dir, 'image')
        self.train_list = osp.join(self.dataset_dir, 'train_test_split/train_list.txt')
        if test_list:
            self.test_list = test_list
        else:
            self.test_list = osp.join(self.dataset_dir, 'train_test_split/test_list_13164.txt')

        required_files = [
            self.dataset_dir,
            self.image_dir,
            self.train_list,
            self.test_list,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_list, is_train=True)
        query, gallery = self.process_dir(self.test_list, is_train=False)

        super(VehicleID, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, list_file, is_train=True):
        img_list_lines = open(list_file, 'r').readlines()

        dataset = []
        for idx, line in enumerate(img_list_lines):
            line = line.strip()
            vid = int(line.split(' ')[1])
            imgid = line.split(' ')[0]
            img_path = osp.join(self.image_dir, imgid + '.jpg')
            if is_train:
                vid = self.dataset_name + "_" + str(vid)
            dataset.append((img_path, vid, int(imgid)))

        if is_train: return dataset
        else:
            random.shuffle(dataset)
            vid_container = set()
            query = []
            gallery = []
            for sample in dataset:
                if sample[1] not in vid_container:
                    vid_container.add(sample[1])
                    gallery.append(sample)
                else:
                    query.append(sample)

            return query, gallery


class SmallVehicleID(VehicleID):
    """VehicleID.
    Small test dataset statistics:
        - identities: 800.
        - images: 6493.
    """

    def __init__(self, root='datasets', **kwargs):
        dataset_dir = osp.join(root, self.dataset_dir)
        self.test_list = osp.join(dataset_dir, 'train_test_split/test_list_800.txt')

        super(SmallVehicleID, self).__init__(root, self.test_list, **kwargs)

class MediumVehicleID(VehicleID):
    """VehicleID.
    Medium test dataset statistics:
        - identities: 1600.
        - images: 13377.
    """

    def __init__(self, root='datasets', **kwargs):
        dataset_dir = osp.join(root, self.dataset_dir)
        self.test_list = osp.join(dataset_dir, 'train_test_split/test_list_1600.txt')

        super(MediumVehicleID, self).__init__(root, self.test_list, **kwargs)

class LargeVehicleID(VehicleID):
    """VehicleID.
    Large test dataset statistics:
        - identities: 2400.
        - images: 19777.
    """

    def __init__(self, root='datasets', **kwargs):
        dataset_dir = osp.join(root, self.dataset_dir)
        self.test_list = osp.join(dataset_dir, 'train_test_split/test_list_2400.txt')

        super(LargeVehicleID, self).__init__(root, self.test_list, **kwargs)


class VeRiWild(ImageDataset):
    """VeRi-Wild.
    Reference:
        Lou et al. A Large-Scale Dataset for Vehicle Re-Identification in the Wild. CVPR 2019.
    URL: `<https://github.com/PKU-IMRE/VERI-Wild>`_
    Train dataset statistics:
        - identities: 30671.
        - images: 277797.
    """
    dataset_dir = "VERI-Wild"
    dataset_name = "veriwild"

    def __init__(self, root='datasets', query_list='', gallery_list='', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.image_dir = osp.join(self.dataset_dir, 'images')
        self.train_list = osp.join(self.dataset_dir, 'train_test_split/train_list.txt')
        self.vehicle_info = osp.join(self.dataset_dir, 'train_test_split/vehicle_info.txt')
        if query_list and gallery_list:
            self.query_list = query_list
            self.gallery_list = gallery_list
        else:
            self.query_list = osp.join(self.dataset_dir, 'train_test_split/test_10000_query.txt')
            self.gallery_list = osp.join(self.dataset_dir, 'train_test_split/test_10000.txt')

        required_files = [
            self.image_dir,
            self.train_list,
            self.query_list,
            self.gallery_list,
            self.vehicle_info,
        ]
        self.check_before_run(required_files)

        self.imgid2vid, self.imgid2camid, self.imgid2imgpath = self.process_vehicle(self.vehicle_info)

        train = self.process_dir(self.train_list)
        query = self.process_dir(self.query_list, is_train=False)
        gallery = self.process_dir(self.gallery_list, is_train=False)

        super(VeRiWild, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, img_list, is_train=True):
        img_list_lines = open(img_list, 'r').readlines()

        dataset = []
        for idx, line in enumerate(img_list_lines):
            line = line.strip()
            vid = int(line.split('/')[0])
            imgid = line.split('/')[1]
            if is_train:
                vid = self.dataset_name + "_" + str(vid)
            dataset.append((self.imgid2imgpath[imgid], vid, int(self.imgid2camid[imgid])))

        assert len(dataset) == len(img_list_lines)
        return dataset

    def process_vehicle(self, vehicle_info):
        imgid2vid = {}
        imgid2camid = {}
        imgid2imgpath = {}
        vehicle_info_lines = open(vehicle_info, 'r').readlines()

        for idx, line in enumerate(vehicle_info_lines[1:]):
            vid = line.strip().split('/')[0]
            imgid = line.strip().split(';')[0].split('/')[1]
            camid = line.strip().split(';')[1]
            img_path = osp.join(self.image_dir, vid, imgid + '.jpg')
            imgid2vid[imgid] = vid
            imgid2camid[imgid] = camid
            imgid2imgpath[imgid] = img_path

        assert len(imgid2vid) == len(vehicle_info_lines) - 1
        return imgid2vid, imgid2camid, imgid2imgpath


class SmallVeRiWild(VeRiWild):
    """VeRi-Wild.
    Small test dataset statistics:
        - identities: 3000.
        - images: 41861.
    """

    def __init__(self, root='datasets', **kwargs):
        dataset_dir = osp.join(root, self.dataset_dir)
        self.query_list = osp.join(dataset_dir, 'train_test_split/test_3000_query.txt')
        self.gallery_list = osp.join(dataset_dir, 'train_test_split/test_3000.txt')

        super(SmallVeRiWild, self).__init__(root, self.query_list, self.gallery_list, **kwargs)


class MediumVeRiWild(VeRiWild):
    """VeRi-Wild.
    Medium test dataset statistics:
        - identities: 5000.
        - images: 69389.
    """

    def __init__(self, root='datasets', **kwargs):
        dataset_dir = osp.join(root, self.dataset_dir)
        self.query_list = osp.join(dataset_dir, 'train_test_split/test_5000_query.txt')
        self.gallery_list = osp.join(dataset_dir, 'train_test_split/test_5000.txt')

        super(MediumVeRiWild, self).__init__(root, self.query_list, self.gallery_list, **kwargs)



class LargeVeRiWild(VeRiWild):
    """VeRi-Wild.
    Large test dataset statistics:
        - identities: 10000.
        - images: 138517.
    """

    def __init__(self, root='datasets', **kwargs):
        dataset_dir = osp.join(root, self.dataset_dir)
        self.query_list = osp.join(dataset_dir, 'train_test_split/test_10000_query.txt')
        self.gallery_list = osp.join(dataset_dir, 'train_test_split/test_10000.txt')

        super(LargeVeRiWild, self).__init__(root, self.query_list, self.gallery_list, **kwargs)
