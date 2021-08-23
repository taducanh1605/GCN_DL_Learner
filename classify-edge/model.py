import torch as th
import torch.nn as nn
import dgl.function as fn
import dgl
import dgl.nn as dglnn
import torch.nn.functional as F

class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            #graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            graph.apply_edges(fn.u_mul_v('h', 'h', 'w_new'))
            #graph.apply_edges(fn.u_add_v('h', 'h', 'sum'))

            #print("------------------------------------------------score:\n", graph.edata['score'][0])
            #print(graph.edata['score'].detach().numpy().shape)

            #print("++++++++++++++++++++++++++++++++++++++++++++++++score:\n", graph.edata['w_new'][0])
            #print(graph.edata['w_new'].detach().numpy().shape)

            #print("================================================score:\n", graph.edata['sum'][0])
            #print(graph.edata['sum'].detach().numpy().shape)
            #label = graph.edata.pop('label')
            #score = graph.edata.pop('score')
            #print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=Before Change:\n",graph.edata['label'].detach().numpy()[:10])
            #return graph.edata.pop('label'), graph.edata['score']
            #return graph.edata['score']
            return graph.edata['w_new']

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h

class BaseRGCN(nn.Module):
    def __init__(self, in_feat, h_dim, out_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, features=None):
        super(BaseRGCN, self).__init__()
        self.in_feat = in_feat
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.features = features

        # create rgcn layers
        self.build_model()

        # create initial features
        if self.features is None:
            self.features = self.create_features()
        
        #self.sage = SAGE(in_feat, num_hidden_layers, out_dim)
        self.pred = DotProductPredictor()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)

        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        return None

    def build_input_layer(self):
        return None

    def build_hidden_layer(self):
        raise NotImplementedError

    def build_output_layer(self):
        return None
    
    def forward(self, g):
        g.ndata['id'] = th.arange(len(g))
        #h = g.nodes()
        #print("++++++++++++++++++++++++++++++++++++++++++\n", g.ndata['h'])
        #print(g.ndata['h'].detach().numpy().shape)
        #print(g.ndata['h'].detach().numpy()[0])
        for layer in self.layers:
            layer(g)
            #h = layer(g)
        h = g.ndata.pop('h')
        #h = self.sage(g, self.features)
        #print("*****************************************\n",h)
        #print(h.detach().numpy().shape)
        #print(h.detach().numpy()[0])
        #print(h.detach().numpy()[0].shape)
        #label, score = self.pred(g, h)
        #return label, score
        return self.pred(g, h)

def initializer(emb):
    emb.uniform_(-1.0, 1.0)
    return emb

class RelGraphEmbedLayer(nn.Module):
    """Embedding layer for featureless heterograph.
    Parameters
    ----------
    storage_dev_id : int
        The device to store the weights of the layer.
    out_dev_id : int
        Device to return the output embeddings on.
    num_nodes : int
        Number of nodes.
    node_tides : tensor
        Storing the node type id for each node starting from 0
    num_of_ntype : int
        Number of node types
    input_size : list of int
        A list of input feature size for each node type. If None, we then
        treat certain input feature as an one-hot encoding feature.
    embed_size : int
        Output embed size
    dgl_sparse : bool, optional
        If true, use dgl.nn.NodeEmbedding otherwise use torch.nn.Embedding
    """
    def __init__(self,
                 storage_dev_id,
                 out_dev_id,
                 num_nodes,
                 node_tids,
                 num_of_ntype,
                 input_size,
                 embed_size,
                 dgl_sparse=False):
        super(RelGraphEmbedLayer, self).__init__()
        self.storage_dev_id = th.device( \
            storage_dev_id if storage_dev_id >= 0 else 'cpu')
        self.out_dev_id = th.device(out_dev_id if out_dev_id >= 0 else 'cpu')
        self.embed_size = embed_size
        self.num_nodes = num_nodes
        self.dgl_sparse = dgl_sparse

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        self.node_embeds = {} if dgl_sparse else nn.ModuleDict()
        self.num_of_ntype = num_of_ntype

        for ntype in range(num_of_ntype):
            if isinstance(input_size[ntype], int):
                if dgl_sparse:
                    self.node_embeds[str(ntype)] = dgl.nn.NodeEmbedding(input_size[ntype], embed_size, name=str(ntype),
                        init_func=initializer, device=self.storage_dev_id)
                else:
                    sparse_emb = th.nn.Embedding(input_size[ntype], embed_size, sparse=True)
                    sparse_emb.cuda(self.storage_dev_id)
                    nn.init.uniform_(sparse_emb.weight, -1.0, 1.0)
                    self.node_embeds[str(ntype)] = sparse_emb
            else:
                input_emb_size = input_size[ntype].shape[1]
                embed = nn.Parameter(th.empty([input_emb_size, self.embed_size],
                                              device=self.storage_dev_id))
                nn.init.xavier_uniform_(embed)
                self.embeds[str(ntype)] = embed

    
    def dgl_emb(self):
        """
        """
        if self.dgl_sparse:
            embs = [emb for emb in self.node_embeds.values()]
            return embs
        else:
            return []

    def forward(self, node_ids, node_tids, type_ids, features):
        """Forward computation
        Parameters
        ----------
        node_ids : tensor
            node ids to generate embedding for.
        node_ids : tensor
            node type ids
        features : list of features
            list of initial features for nodes belong to different node type.
            If None, the corresponding features is an one-hot encoding feature,
            else use the features directly as input feature and matmul a
            projection matrix.
        Returns
        -------
        tensor
            embeddings as the input of the next layer
        """
        embeds = th.empty(node_ids.shape[0], self.embed_size, device=self.out_dev_id)

        # transfer input to the correct device
        type_ids = type_ids.to(self.storage_dev_id)
        node_tids = node_tids.to(self.storage_dev_id)

        # build locs first
        locs = [None for i in range(self.num_of_ntype)]
        for ntype in range(self.num_of_ntype):
            locs[ntype] = (node_tids == ntype).nonzero().squeeze(-1)
        for ntype in range(self.num_of_ntype):
            loc = locs[ntype]
            if isinstance(features[ntype], int):
                if self.dgl_sparse:
                    embeds[loc] = self.node_embeds[str(ntype)](type_ids[loc], self.out_dev_id)
                else:
                    embeds[loc] = self.node_embeds[str(ntype)](type_ids[loc]).to(self.out_dev_id)
            else:
                embeds[loc] = features[ntype][type_ids[loc]].to(self.out_dev_id) @ self.embeds[str(ntype)].to(self.out_dev_id)

        return embeds