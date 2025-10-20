# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from abc import ABC, abstractmethod

import numpy as np

from . import functional as F


class DataPartitioner(ABC):
    """Base class for data partition in federated learning.

    Examples of :class:`DataPartitioner`: :class:`BasicPartitioner`, :class:`CIFAR10Partitioner`.

    Details and tutorials of different data partition and datasets, please check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.
    """

    def __init__(self):
        pass

    @abstractmethod
    def _perform_partition(self):
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()


class BasicPartitioner(DataPartitioner):
    """Basic data partitioner.

    Basic data partitioner, supported partition:

    - label-distribution-skew:quantity-based

    - label-distribution-skew:distributed-based (Dirichlet)

    - quantity-skew (Dirichlet)

    - IID

    For more details, please check `Federated Learning on Non-IID Data Silos: An Experimental Study <https://arxiv.org/abs/2102.02079>`_ and `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.

    Args:
        targets (list or numpy.ndarray): Sample targets. Unshuffled preferred.
        num_clients (int): Number of clients for partition.
        partition (str): Partition name. Only supports ``"noniid-#label"``, ``"noniid-labeldir"``, ``"unbalance"`` and ``"iid"`` partition schemes.
        dir_alpha (float): Parameter alpha for Dirichlet distribution. Only works if ``partition="noniid-labeldir"``.
        major_classes_num (int): Number of major class for each clients. Only works if ``partition="noniid-#label"``.
        verbose (bool): Whether output intermediate information. Default as ``True``.
        min_require_size (int, optional): Minimum required sample number for each client. If set to ``None``, then equals to ``num_classes``. Only works if ``partition="noniid-labeldir"``.
        seed (int): Random seed. Default as ``None``.

    Returns:
        dict: ``{ client_id: indices}``.
    """

    def __init__(self, targets, num_clients, num_classes,
                 partition='iid',
                 dir_alpha=None,
                 major_classes_num=1,
                 verbose=True,
                 min_require_size=None,
                 seed=None):
        self.targets = np.array(targets)  # with shape (num_samples,)
        self.num_samples = self.targets.shape[0]
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.client_dict = dict()
        self.partition = partition
        self.dir_alpha = dir_alpha
        self.verbose = verbose
        self.min_require_size = min_require_size

        # self.rng = np.random.default_rng(seed)  # rng currently not supports randint
        np.random.seed(seed)

        if partition == "noniid-#label":
            # label-distribution-skew:quantity-based
            assert isinstance(major_classes_num, int), f"'major_classes_num' should be integer, " \
                                                       f"not {type(major_classes_num)}."
            assert major_classes_num > 0, f"'major_classes_num' should be positive."
            assert major_classes_num < self.num_classes, f"'major_classes_num' for each client " \
                                                         f"should be less than number of total " \
                                                         f"classes {self.num_classes}."
            self.major_classes_num = major_classes_num
        elif partition in ["noniid-labeldir", "unbalance"]:
            # label-distribution-skew:distributed-based (Dirichlet) and quantity-skew (Dirichlet)
            assert dir_alpha > 0, f"Parameter 'dir_alpha' for Dirichlet distribution should be " \
                                  f"positive."
        elif partition == "iid":
            # IID
            pass
        else:
            raise ValueError(
                f"tabular data partition only supports 'noniid-#label', 'noniid-labeldir', "
                f"'unbalance', 'iid'. {partition} is not supported.")

        self.client_dict = self._perform_partition()
        # get sample number count for each client
        self.client_sample_count = F.samples_num_count(self.client_dict, self.num_clients)
        self.stats_report = F.partition_report(targets, self.client_dict, class_num=self.num_classes, verbose=False)

    def _perform_partition(self):
        if self.partition == "noniid-#label":
            # label-distribution-skew:quantity-based
            client_dict = F.label_skew_quantity_based_partition(self.targets, self.num_clients,
                                                                self.num_classes,
                                                                self.major_classes_num)

        elif self.partition == "noniid-labeldir":
            # label-distribution-skew:distributed-based (Dirichlet)
            client_dict = F.hetero_dir_partition(self.targets, self.num_clients, self.num_classes,
                                                 self.dir_alpha,
                                                 min_require_size=self.min_require_size)

        elif self.partition == "unbalance":
            # quantity-skew (Dirichlet)
            client_sample_nums = F.dirichlet_unbalance_split(self.num_clients, self.num_samples,
                                                             self.dir_alpha)
            client_dict = F.homo_partition(client_sample_nums, self.num_samples)

        else:
            # IID
            client_sample_nums = F.balance_split(self.num_clients, self.num_samples)
            client_dict = F.homo_partition(client_sample_nums, self.num_samples)

        return client_dict

    def __getitem__(self, index):
        return self.client_dict[index]

    def __len__(self):
        return len(self.client_dict)


def create_noniid_users(data_content, args, alpha=100):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :param alpha:
    :return:
    """
    targets = data_content['train']['labels'] + data_content['val']['labels'] + data_content['test']['labels']
    if args.data == 'ag_news':
        num_classes = 4
    else:
        num_classes = 2

    alpha /= num_classes

    if isinstance(targets[0], list):
        targets = [t[0] for t in targets]
    targets_dict = dict(zip(set(targets), range(len(set(targets)))))
    targets = [targets_dict[t] for t in targets]

    data_partitioner = BasicPartitioner(targets,
                                        num_clients=args.num_clients,
                                        partition="noniid-labeldir",
                                        verbose=False,
                                        dir_alpha=alpha,
                                        num_classes=num_classes)

    dict_users = data_partitioner.client_dict

    train_dict = {k: [v for v in d if v < len(data_content['train'])] for k, d in dict_users.items()}
    val_dict = {k: [v - len(data_content['train']) for v in d if
                    len(data_content['train']) + len(data_content['val']) > v >= len(data_content['train'])] for k, d in
                dict_users.items()}
    test_dict = {k: [v - len(data_content['train']) - len(data_content['val']) for v in d if
                     v >= len(data_content['train']) + len(data_content['val'])] for k, d in dict_users.items()}

    stats_df = F.partition_report(targets, dict_users, class_num=num_classes, verbose=True)
    return train_dict, val_dict, test_dict, stats_df
