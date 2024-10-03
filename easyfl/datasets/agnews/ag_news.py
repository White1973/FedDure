# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""AG News topic classification dataset."""

import pdb
import csv
import os
import datasets
from datasets.tasks import TextClassification
import numpy as np
from easyfl.datasets.utils.base_dataset import BaseDataset
from dataset import load_dataset

_DESCRIPTION = """\
AG is a collection of more than 1 million news articles. News articles have been
gathered from more than 2000 news sources by ComeToMyHead in more than 1 year of
activity. ComeToMyHead is an academic news search engine which has been running
since July, 2004. The dataset is provided by the academic comunity for research
purposes in data mining (clustering, classification, etc), information retrieval
(ranking, search, etc), xml, data compression, data streaming, and any other
non-commercial activity. For more information, please refer to the link
http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html .

The AG's news topic classification dataset is constructed by Xiang Zhang
(xiang.zhang@nyu.edu) from the dataset above. It is used as a text
classification benchmark in the following paper: Xiang Zhang, Junbo Zhao, Yann
LeCun. Character-level Convolutional Networks for Text Classification. Advances
in Neural Information Processing Systems 28 (NIPS 2015).
"""

_CITATION = """\
@inproceedings{Zhang2015CharacterlevelCN,
  title={Character-level Convolutional Networks for Text Classification},
  author={Xiang Zhang and Junbo Jake Zhao and Yann LeCun},
  booktitle={NIPS},
  year={2015}
}
"""

_TRAIN_DOWNLOAD_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
_TEST_DOWNLOAD_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"


# class AGNews:
#     """AG News topic classification dataset."""
#
#     def _info(self):
#         return datasets.DatasetInfo(
#             description=_DESCRIPTION,
#             features=datasets.Features(
#                 {
#                     "text": datasets.Value("string"),
#                     "label": datasets.features.ClassLabel(names=["World", "Sports", "Business", "Sci/Tech"]),
#                 }
#             ),
#             homepage="http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html",
#             citation=_CITATION,
#             task_templates=[TextClassification(text_column="text", label_column="label")],
#         )
#
#     def _split_generators(self, dl_manager):
#         train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
#         test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
#         return [
#             datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
#             datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
#         ]
#
    # def _generate_examples(self, filepath):
    #     """Generate AG News examples."""
    #     with open(filepath, encoding="utf-8") as csv_file:
    #         csv_reader = csv.reader(
    #             csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
    #         )
    #         for id_, row in enumerate(csv_reader):
    #             label, title, description = row
    #             # Original labels are [1, 2, 3, 4] ->
    #             #                   ['World', 'Sports', 'Business', 'Sci/Tech']
    #             # Re-map to [0, 1, 2, 3].
    #             label = int(label) - 1
    #             text = " ".join((title, description))
    #             yield id_, {"text": text, "label": label}

AGNEWS='agnews'
class Agnews(BaseDataset):
    def __init__(self,
                 root,
                 fraction,
                 split_type,
                 user,
                 iid_user_fraction=0.1,
                 train_test_split=0.9,
                 minsample=10,
                 num_class=80,
                 num_of_client=100,
                 class_per_client=2,
                 setting_folder=None,
                 seed=-1,
                 weights=None,
                 alpha=0.5,
                 ssl_senario='server',
                 num_labels_per_class=5,
                 is_ssl=False,local_test=None,s_split_type=None):
        super(Agnews, self).__init__(root,
                                      AGNEWS,
                                      fraction,
                                      split_type,
                                      user,
                                      iid_user_fraction,
                                      train_test_split,
                                      minsample,
                                      num_class,
                                      num_of_client,
                                      class_per_client,
                                      setting_folder,
                                      seed)
        self.train_data, self.test_data = {}, {}
        self.split_type = split_type
        self.num_of_client = num_of_client
        self.weights = weights
        self.alpha = alpha
        self.min_size = minsample
        self.class_per_client = class_per_client
        self.ssl_senario = ssl_senario
        self.num_labels_per_class = num_labels_per_class
        self.is_ssl = is_ssl
        self.local_test=local_test
        self.s_split_type=s_split_type

    def extract_data(self, data):
        texts, labels = [], []
        for title, description, label in zip(data['Title'], data['Description'], data['Class Index']):
            label = int(label) - 1
            text = " ".join((title, description))
            texts.append(text)
            labels.texts.append(label)

        return np.array(texts), np.array(labels)

    def download_raw_file_and_extract(self):
        self.dataset = load_dataset('ag_news')

        train_data, train_label = self.extract_data(self.dataset['train'])
        test_data, test_label = self.extract_data(self.dataset['test'])



    def preprocess(self):
        pass


if __name__=="__main__":
    pass
    #agNews = AGNews()
    # path = '/mnt/lustre/share_data/qiuzengyu/fssl_data/agnews'
    # train_path = os.path.join(path, 'train.csv')
    # test_path = os.path.join(path, 'test.csv')
    # train_set = agNews._generate_examples(train_path)
    # test_set = agNews._generate_examples(test_path)
    # pdb.set_trace()

