import collections
import io
import multiprocessing
import os
import random
import zipfile

from absl import app
from absl import flags
import networkx as nx
import numpy as np
from scipy.spatial import distance
from six.moves import urllib
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
import tqdm

import DDGK4G.model as model

FLAGS = flags.FLAGS
flags.DEFINE_string("data_set",None,"The data set.")
flags.DEFINE_integer("num_sources", 16, "The number of source graphs.")
flags.DEFINE_integer("num_threads", 32, "The number of threads")
flags.DEFINE_string("working_dir", None, "The working directory")

def load():
    resp = urllib.request.urlopen(FLAGS.data_set)
    unzipped = zipfile.ZipFile(io.BytesIO(resp.read()))
    g=nx.Graph()
    
    def path(suffix):
        paths =[n for n in unzipped.namelist() if n.endswith(suffix)]
        assert len(paths) <=1
        return paths[0] if paths else None

    def open(suffix):
        return unzipped.open(path(suffix),'r')'
    
    try:
        for i, line in enumerate(open('_node_labels.txt'),1):
            g.add_node(i, label=int(line))
    except KeyError:
        pass