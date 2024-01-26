import networkx as nx
import numpy as np
import tensorflow as tf
from collections import namedtuple

_USE_GLOBAL_STEP=0

def MutagHParams(embedding_size=4, num_dnn_layers=4, score_window=10, learning_rate=0.01, train_num_epochs=600, score_num_epochs=600, node_label_loss_coefficient=0.0, num_node_labels=7, incident_label_loss_coefficient=0.0, num_edge_labels=4):
    """
    Returns HParams object which stores the hyperparameters for DDGK model used for the MUTAG benchmark dataset
        Args:
            embedding_size (int): The size of node embeddings
            num_dnn_layers (int): The number of layers in the DNN
            score_window (int): The window to average for scoring loss and accuracy calculation.
            learning_rate (int): The adam learning rate
            train_num_epochs (int): The steps for training
            score_num_epochs (int): The steps for node mapping and scoring
            node_label_loss_coefficient (float): Label preserving loss for node mapping
            num_node_labels (int): The number of node labels
            incident_label_loss_coefficient (float): Label preserving loss for incident edges
            num_edge_labels (int): The number of edge labels
            
        Returns:
            Named Tuple HParams: Hyperparameter set for DDGK
    """
    return namedtuple('HParams',MutagHParams.__code__.co_varnames)(MutagHParams.__defaults__)

def AdjMatrixAccuracy(logits, labels):
    """
    Returns Adjacency Matrix wise accuracy, or accuracy per edge
        Args:
            logits (tf.Tensor): Logits tensor, containing the predicted values for edges
            labels (tf.Tensor): Labels tensor, containing the actual values for edges
        Returns:
            tf.Tensor: Reduced tensor containing the edge-wise accuracies
    """
    predictions = tf.cast(tf.greater(tf.sigmoid(logits),0.5), tf.float64)
    accuracies = tf.cast(tf.equal(predictions, labels), tf.float64)
    
    return tf.reduce_mean(accuracies) #Reports accuracy per edge
def LogitsFromProb(prob):
    """
    Calculates logits from probabilities
        Args:
            prob (tf.Tensor): Probability tensor
        Returns:
            tf.Tensor: Logits tensor
    """
    return tf.log(tf.clip_by_value(prob, 1e-12, 1.0))
def ProbFromCounts(counts):
    """
    Returns Probability Matrix from count matrix of events
        Args:
            counts (tf.Tensor): Count tensor
        Returns:
            tf.Tensor: Probabilities Tensor
    """
    return counts/ tf.clip_by_value(
        tf.reduce_sum(counts, axis=1, keepdims=True), 1e-9, 1e9
    )
def AdjMatrixLoss(logits, labels):
    """
    Returns Loss matrix calculated from Logit vs Label Adjacency matrices
        Args:
            logits (tf.Tensor): Logit Tensor or Predicted Adjacency Matrix
            labels (tf.Tensor): Label Tensor or Actual Adjacency Matrix 
        Returns:
            tf.Tensor: Sigmoid Cross Entropy Losses Matrix
    """
    losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(losses)
def NodesLabels(graph, num_labels):
    """
    Returns the probability matrix of Node Labels
        Args:
            graph (nx.Graph): Graph
            num_labels (int): Number of node label types
        Returns:
            tf.Tensor: Probabilities tensor of nodes
    """
    #labels size is (graph_num_node, 1)
    labels = [graph.nodes[i]['Label'] for i in graph.nodes()]
    #labels size is (graph_num_node, num_labels)
    labels = tf.one_hot(labels, num_labels, dtype=tf.float64)
    return ProbFromCounts(labels)
def NeighborNodesLabels(graph, num_labels):
    """
    Returns the probability matrix of Neighborhood Node Labels
        Args:
            graph (nx.Graph): Graph
            num_labels (int): Number of node label types
        Returns:
            tf.Tensor: Probabilities tensor of Neighbor nodes
    """
    Neighbors_labels =  np.zeros((graph.number_of_nodes(), num_labels))
    for v in graph.nodes():
        for u in graph.Neighbors(v):
            Neighbors_labels[v, graph.nodes[u]['label']] += 1.0
            return ProbFromCounts(Neighbors_labels)
def EdgesLabels(graph, num_labels):
    """
    Returns the probability matrix of Edge Labels
        Args:
            graph (nx.Graph): Graph
            num_labels (int): Number of edge label types
        Returns:
            tf.Tensor: Probabilities tensor of edges
    """
    labels = np.zeros((graph.number_of_nodes(), num_labels))
    
    for i, n in enumerate(graph.nodes()):
        for u,v in graph.edges(n):
            labels[i, graph[u][v]['label']] += 1.0

    return ProbFromCounts(labels)
def NeighborEdgesLabels(graph, num_labels):
    """
    Returns the probability matrix of Neighboring Edge Labels
        Args:
            graph (nx.Graph): Graph
            num_labels (int): Number of edge label types
        Returns:
            tf.Tensor: Probabilities tensor of edges
    """
    labels = np.zeros((graph.number_of_nodes(), num_labels))
    for i,v in enumerate(graph.nodes()):
        for u in graph.neighbours(v):
            for v1, v2, d in graph.edges(u, data=True):
                if v not in (v1,v2):
                    labels[i, d['label']] += 1.0
    return ProbFromCounts(labels)
def NodeLabelLoss(source, source_node_prob, target, num_labels):
    """
    Returns the losses matrix for node labels
        Args:
            source (tf.Tensor): Predicted Node label tensor
            source_node_prob (tf.Tensor): Predicted Node label probability tensor
            target (tf.Tensor): Node Target label tensor
            num_labels (int): Number of node labels
        Returns:
            tf.Tensor: Reduced mean tensor representing the Softmax Cross entropy losses
    """
    #source_labels size is (source_num_node, num_labels)
    source_labels = NodesLabels(source, num_labels)
    #target_labels_size is (target_num_node, num_labels)
    target_labels = NodesLabels(target, num_labels)
    #Logarithm is found because result of multiplication is 
    #probability.
    logits = LogitsFromProb(tf.matmul(source_node_prob, source_labels))
    losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_labels, logits=logits)
    
    return tf.reduce_mean(losses)
def NeighborNodeLabelLoss(target_neighbors_prob, target, num_labels):
    """
    Returns the losses matrix for neighbor node labels
        Args:
            target_neighbors_prob (tf.Tensor): Target Neighbor Probability Matrix
            target (tf.Tensor): Target Node matrix
            num_labels (int): Number of node labels
        Returns:
            tf.Tensor: Reduced mean tensor representing the Softmax Cross entropy losses
    """
    target_labels = NodesLabels(target, num_labels)
    target_neighbors_labels = NeighborNodesLabels(target, num_labels)
    logits = LogitsFromProb(tf.matmul(target_neighbors_prob, target_labels))
    losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=target_neighbors_labels, logits=logits
    )
    return tf.reduce_mean(losses)
def EdgeLabelLoss(source, source_node_prob, target, num_labels):
    """
    Returns the losses matrix for edge labels
        Args:
            source (tf.Tensor): Predicted Edge label tensor
            source_node_prob (tf.Tensor): Predicted Edge label probability tensor
            target (tf.Tensor): Edge Target label tensor
            num_labels (int): Number of edge labels
        Returns:
            tf.Tensor: Reduced mean tensor representing the Softmax Cross entropy losses
    """
    source_labels = EdgesLabels(source, num_labels)
    target_labels = EdgesLabels(target, num_labels)
    logits = LogitsFromProb(tf.matmul(source_node_prob, source_labels))
    losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=target_labels, logits=logits
    )
    return tf.reduce_mean(losses)
def NeighborEdgeLabelsLoss(target_neighbors_prob, target, num_labels):
    """
    Returns the losses matrix for neighbor edge labels
        Args:
            target_neighbors_prob (tf.Tensor): Target Neighbor Probability Matrix
            target (tf.Tensor): Target Edge matrix
            num_labels (int): Number of edge labels
        Returns:
            tf.Tensor: Reduced mean tensor representing the Softmax Cross entropy losses
    """
    target_labels = EdgesLabels(target, num_labels)
    target_neighbors_labels = NeighborEdgesLabels(target, num_labels)
    logits = LogitsFromProb(tf.matmul(target_neighbors_prob, target_labels))
    losses = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=target_neighbors_labels, logits = logits
    )
    return tf.reduce_mean(losses)

def create_train_op(total_loss,
                    optimizer,
                    global_step=_USE_GLOBAL_STEP,
                    update_ops=None,
                    variables_to_train=None,
                    transform_grads_fn=None,
                    summarize_gradients=False,
                    aggregation_method=None,
                    colocate_gradients_with_ops=False,
                    check_numerics=True):
  """
  Creates an `Operation` that evaluates the gradients and returns the loss.
    Args:
        total_loss: A `Tensor` representing the total loss.
        optimizer: A tf.Optimizer to use for computing the gradients.
        summarize_gradients: Whether or not add summaries for each gradient.

    Returns:
        A `Tensor` that when evaluated, computes the gradients and returns the total
        loss value.
  """
  if global_step is _USE_GLOBAL_STEP:
    global_step = tf.training_util.get_or_create_global_step()

  # Update ops use GraphKeys.UPDATE_OPS collection if update_ops is None.
  global_update_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS))
  if update_ops is None:
    update_ops = global_update_ops
  else:
    update_ops = set(update_ops)
  if not global_update_ops.issubset(update_ops):
    logging.warning('update_ops in create_train_op does not contain all the '
                    'update_ops in GraphKeys.UPDATE_OPS')

  # Make sure update_ops are computed before total_loss.
  if update_ops:
    with ops.control_dependencies(update_ops):
      barrier = control_flow_ops.no_op(name='update_barrier')
    total_loss = control_flow_ops.with_dependencies([barrier], total_loss)

  if variables_to_train is None:
    # Default to tf.compat.v1.trainable_variables()
    variables_to_train = tf_variables.trainable_variables()
  else:
    # Make sure that variables_to_train are in
    # tf.compat.v1.trainable_variables()
    for v in variables_to_train:
      assert v.trainable or v in tf_variables.trainable_variables()

  assert variables_to_train

  # Create the gradients. Note that apply_gradients adds the gradient
  # computation to the current graph.
  grads = optimizer.compute_gradients(
      total_loss,
      variables_to_train,
      gate_gradients=gate_gradients,
      aggregation_method=aggregation_method,
      colocate_gradients_with_ops=colocate_gradients_with_ops)

  if transform_grads_fn:
    grads = transform_grads_fn(grads)

  # Summarize gradients.
  if summarize_gradients:
    with ops.name_scope('summarize_grads'):
      add_gradients_summaries(grads)

  # Create gradient updates.
  grad_updates = optimizer.apply_gradients(grads, global_step=global_step)

  with ops.name_scope('train_op'):
    # Make sure total_loss is valid.
    if check_numerics:
      total_loss = array_ops.check_numerics(total_loss,
                                            'LossTensor is inf or nan')

    # Ensure the train_tensor computes grad_updates.
    train_op = control_flow_ops.with_dependencies([grad_updates], total_loss)

  # Add the operation used for training to the 'train_op' collection
  train_ops = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
  if train_op not in train_ops:
    train_ops.append(train_op)

  return train_op

def Encode(source, ckpt_prefix, hparams):
    """
    Runs encoder and saves its trainable parameters.
        Args:
            source (tf.Graph): Source Graph
            ckpt_prefix (tf.train.Checkpoint): Tensorflow Checkpoint prefix
            hparams (namedtuple "HParams"): Hyperparameter set
    """
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.reset_default_graph()
    
    g=tf.Graph()
    session = tf.Session(graph=g)
    
    with g.as_default(), session.as_default():
        A = nx.adjacency_matrix(source, weight=None)
        x = tf.one_hot(
            list(source.nodes()), source.number_of_nodes(), dtype=tf.float64)
        y = tf.convert_to_tensor(A.todense(), dtype = tf.float64)
        
        layer = tf.layers.dense(x, hparams.embedding_size, use_bias=True)
        for _ in range(hparams.num_dnn_layers):
            layer = tf.layers.dense(
                layer, hparams.embedding_size * 4, activation=tf.nn.tanh
            )
        
        logits = tf.layers.dense(
            layer, source.number_of_nodes(), activation=tf.nn.tanh)
        loss = AdjMatrixLoss(logits, y)
        
        train_op = create_train_op(
            loss,
            tf.train.AdamOptimizer(hparams.learning_rate),
            summarize_gradients=False
        )
        session.run(tf.global_variables_initializer)
        
        for _ in range(hparams.train_num_epochs):
            session.run(train_op)
        
        tf.train.Saver(tf.trainable_variables()).save(session, ckpt_prefix)
def Score(source, target, ckpt_prefix, hparams):
    """
    Calculates the divergence differences between Source graph and Target graph
    Args:
        source (nx.Graph): Source Graph
        target (nx.Graph): Target Graph
        ckpt_prefix (tf.train.Checkpoint): Tensorflow Checkpoint prefix
        hparams (tf.contrib.contrib_training.HParams): Hyperparameter set
    Returns:
        list: Losses of Source graph vs Target graph
    """
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.reset_default_graph()
    
    g=tf.Graph()
    session = tf.Session(graph=g)
    
    with g.as_default(), session.as_default():
        A=nx.adjacency_matrix(target, weight=None)
        x=tf.one_hot(
            list(target.nodes()), target.number_of_nodes(), dtype=tf.float64
        )
        y=tf.convert_to_tensor(A.todense(), dtype=tf.float64)
        
        with tf.variable_scope('attention'):
            attention = tf.layers.dense(x, source.number_of_nodes, use_bias=False)
            source_node_prob = tf.nn.softmax(attention)
            
        layer = tf.layers.dense(
            source_node_prob, hparams.embedding_size, use_bias=False
        )
        for _ in range(hparams.num_dnn_layers):
            layer = tf.layers.dense(
                layer, hparams.embedding_size * 4, activation=tf.nn.tanh
            )
        
        logits = tf.layers.dense(
            source_node_prob, hparams.embedding_size, use_bias=False
        )
        
        with tf.variable_scope('attention_reverse'):
            attention_reverse = tf.layers.dense(logits, target.number_of_nodes())
            target_neighbors_pred = tf.nn.sigmoid(attention_reverse)
            target_neighbors_prob = ProbFromCounts(target_neighbors_prob)
            
        loss = AdjMatrixLoss(attention_reverse, y)
        
        if hparams.get('incident_label_loss_coefficient', None):
            label_loss = NodeLabelLoss(source, source_node_prob, target, hparams.num_node_labels)
            label_loss += NeighborNodeLabelLoss(target_neighbors_prob, target,
                                                 hparams.num_node_labels)
            loss += label_loss * hparams.node_label_loss_coefficient
            
        if hparams.get('incident_label_loss_coefficient', None):
            edge_loss = EdgeLabelLoss(source, source_node_prob, target,
                                      hparams.num_edge_labels)
            
            edge_loss += NeighborEdgeLabelsLoss(target_neighbors_prob, target,
                                                hparams.num_edge_labels)
            
            loss += edge_loss * hparams.incident_label_loss_coefficient
            
        vars_to_restore = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='(?!attention)'
        )
        vars_to_train = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='attention'
        )
        
        train_op = contrib_training.create_train_op(
            loss,
            tf.train.AdamOptimizer(hparams.learning_rate),
            variables_to_train=vars_to_train,
            summarize_gradients=False
        )
        
        session.run(tf.global_variables_initalizer())
        
        tf.train.Saver(vars_to_restore).restore(session, ckpt_prefix)
        
        losses = []
        
        for _ in range(hparams.score_num_epochs):
            losses.append(session.run([train_op, loss])[1])
        
    return losses[-hparams.score_window:]