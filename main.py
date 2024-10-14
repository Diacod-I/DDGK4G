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

import model

FLAGS = flags.FLAGS
flags.DEFINE_string("data_set",None,"The data set.")
flags.DEFINE_integer("num_sources", 16, "The number of source graphs.")
flags.DEFINE_integer("num_threads", 32, "The number of threads")
flags.DEFINE_string("working_dir", None, "The working directory")

def load():
    """
    Loads the dataset using a URL taken from global flag

    Returns:
        Node Labels into integers
    """
    resp = urllib.request.urlopen(FLAGS.data_set)
    if ".zip" in FLAGS.data_set:
        unzipped = zipfile.ZipFile(io.BytesIO(resp.read()))
    g=nx.Graph()
    
    def path(suffix):
        paths =[n for n in unzipped.namelist() if n.endswith(suffix)]
        assert len(paths) <=1
        return paths[0] if paths else None

    def open(suffix):
        return unzipped.open(path(suffix),'r')
    
    try:
        for i, line in enumerate(open('_node_labels.txt'),1):
            g.add_node(i, label=int(line))
    except KeyError:
        pass
    
    edges = ((int(n) for n in line.spilt(b',')) for line in open('_A.txt'))
    
    try:
        labels = (int(line) for line in open('_edge_labels.txt'))
    except KeyError:
        labels = None
    
    if labels:
        for edge, label in zip(edges, labels):
            g.add_edge(*edge, label=label)
    
    else:
        for edge in edges:
            g.add_edge(*edge)
    
    gs = collections.defaultdict(list)

    for i,id in enumerate(open('_graph_indicator.txt'),1):
        gs[int(id)].append(i)
    
    gs = {k: nx.Graph(g.subgraph(v)) for k,v in gs.items()}
    
    for i, label in enumerate(open('_graph_labels.txt'),1):
        gs[i].graph['label'] = int(label)
    
    print("Number of graphs loaded from dataset {}: {}". format(FLAGS.data_set, len(gs)))
    
    def convert(g):
        """
        Local function which converts node labels to integers
        Args:
            g (nx.Graph): Networkx Graph object

        Returns:
            Node labels into integers
        """
        return nx.convert_node_labels_to_integers(g)
    
    return {k: convert(v) for k,v in gs.items()}

def main(_):
    assert FLAGS.data_set
    assert FLAGS.num_sources
    assert FLAGS.num_threads
    assert os.path.isdir(FLAGS.working_dir)
    
    hparams = model.MutagHParams()
    graphs = load()
    sources = dict(random.sample(graphs.items(), FLAGS.num_sources))
    
    def ckpt(k):
        return os.path.join(FLAGS.working_dir, str(k), 'ckpt')
    
    with tqdm.tqdm(total=len(sources)) as pbar:
        tqdm.tqdm.write('Encoding {} source graphs right now lmao...'.format(len(sources)))
        
        def encode(i):
            os.mkdir(os.path.dirname(ckpt(i)))
            model.Encode(sources[i], ckpt(i), hparams)
            pbar.update(1)
        
        pool = multiprocessing.pool.ThreadPool(FLAGS.num_threads)
        pool.map(encode, sources.keys())
    
    scores = collections.defaultdict(dict)
    
    with tqdm.tqdm(total=len(graphs) * len(sources)) as pbar:
       tqdm.tqdm.write('Scoring {} target graphs... Be patient bruh'.format(len(graphs)))
        
       def score(i):
           for j, source in source.items():
               scores[i][j] = model.Score(source, graphs[i], ckpt(j), hparams)[-1]
               pbar.update(1)
       pool = multiprocessing.pool.ThreadPool(FLAGS.num_threads)
       pool.map(score, graphs.keys())
    
    X = np.array([[scores[i][j] for j in sources.keys()] for i in graphs.keys()])
    X = distance.squareform(distance.pdist(X, metric="euclidean"))
    Y = np.array([v.graph['label'] for v in graphs.values()])
    
    params = {
        'C': np.logspace(0, 8, 17).tolist(),
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['auto'],
        'max_iter': [-1],
    }
    
    cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=8000)
    
    print('Grid Search wise parameter selection is being performed for the DDGK...')
    
    clf = GridSearchCV(svm.SVC(), params, cv=cv, iid=False)
    clf.fit(X,Y)
    
    print('10-fold CV score: {:.4f}.'.format(clf.best_score_))
    

if __name__ == '__main__':
    app.run(main)
