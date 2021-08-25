import dgl.function as fn
from dgl import DGLGraph
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import RGCNBasisLayer as RGCNLayer
from model import BaseRGCN
from utils import _select_threshold, metrics
import warnings
import os
from owlready2 import *
from gensim.models import KeyedVectors
import re
from num2words import num2words
from sklearn.model_selection import train_test_split
import csv
import pandas as pd
from random import randint
from dataProcessing import *
import itertools as it



class EdgeClassify(BaseRGCN):
    def create_features(self):
        features = torch.arange(self.in_feat)
        if self.use_cuda:
            featurs = features.cuda()
        return features

    def build_input_layer(self):
        return RGCNLayer(self.in_feat, self.h_dim, self.num_rels, self.num_bases,
                         activation=F.relu, is_input_layer=True, node_features=self.features)

    def build_hidden_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases, dropout=self.dropout,
                         activation=F.relu)

    def build_output_layer(self):
        return RGCNLayer(self.h_dim, self.out_dim, self.num_rels, self.num_bases,
                         activation=None)

def evaluate(model, graph, labels, mask):
    model.eval()
    with torch.no_grad():
        logits, _ = model(graph)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def main(args):
    fold = 1
    use_cuda = args.gpu
    
    #manager input and output folder
    path = args.dataset + "/"
    pathData = "../rdf_xml_dataset/"
    dataset = pathData + args.dataset + ".owl"
    if not os.path.exists(path):
        os.mkdir(path)
    ontology = get_ontology(dataset)
    
    #fold loop
    for i in range(fold):

        outputPath = path + "fold_" + str(i) + "/"
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)
        newDataset = outputPath + args.dataset + "_new_" + str(args.n_hidden) +".owl"
        outputFile = outputPath + "results_" + str(args.n_hidden) + ".txt"

        #------------------------data processing
        onto = ontology.load()
        with onto:
            #sync_reasoner()
            sync_reasoner_pellet(infer_property_values = True, infer_data_property_values = True, debug = 9)
        indList = list(onto.individuals())
        num_node = len(indList)
        node_idx = list(range(0,len(indList)))
        classList = list(onto.classes())
        objPropList = list(onto.object_properties())
        edgeList = np.array(edge_list1(objPropList, indList, onto), dtype=np.int) #[src,dst,type]
        edgeIdx = list(range(0,len(edgeList)))

        print("****************************** Fold", i,"******************************")
        output = ""
        print("number of relations: ", len(edgeList))
        #------------------------init dictionary and edge_type
        edge_dict = []
        edge_label = []
        edge_connect_temp = list(it.permutations(node_idx, 2))
        for e in edge_connect_temp:
            edge_dict.append([e[0],e[1]])
            edge_label.append([0]*len(objPropList))
        edge_label = np.array(edge_label)
        edge_type = np.array([0]*len(edge_label))

        #------------------------split train/test set
        rand_split = randint(1,99)
        edgeTrainIdx, edgeTestIdx, edgeTrain, edgeTest = train_test_split(edgeIdx, edgeList, test_size=0.1, random_state=rand_split)

        #------------------------make label for trainSet
        #------------------------make edge_type (0: not sure, 1: in dataset)
        true_mask = torch.zeros(len(edge_dict), dtype=torch.bool)
        for e in edgeTrain:
            if [e[0],e[1]] in edge_dict:
                eidx = edge_dict.index([e[0],e[1]])
                edge_label[eidx][e[2]] = 1
                true_mask[eidx] = True
                edge_type[eidx] = 1

        #make train/validation mask
        train_valid_mask = torch.zeros(len(edge_label[true_mask]), dtype=torch.bool).bernoulli(0.9)
        train_mask = torch.zeros(len(edge_dict), dtype=torch.bool)
        val_mask = torch.zeros(len(edge_dict), dtype=torch.bool)
        count = 0
        for x in range(len(true_mask)):
            if true_mask[x] == True:
                if train_valid_mask[count] == True:
                    train_mask[x] = True
                else:
                    val_mask[x] = True
                count += 1

        #print(train_dict[:20])
        #train_dict = np.array(train_dict, dtype=np.int)
        #print(train_dict[:20])

        #------------------------make test_mask
        test_mask = torch.zeros(len(edge_dict), dtype=torch.bool)
        for e in edgeTest:
            if [e[0],e[1]] in edge_dict:
                eidx = edge_dict.index([e[0],e[1]])
                test_mask[eidx] = True
        
        #------------------------make test_label
        test_label = edge_label[test_mask]
        edge_test = np.array(edge_dict)[test_mask].tolist()
        for e in edgeTest:
            if [e[0],e[1]] in edge_dict:
                eidx = edge_test.index([e[0],e[1]])
                test_label[eidx, e[2]] = 1


        #get src, dst, label
        edge_src, edge_dst = np.array(edge_dict).transpose()
        num_rel = 2
        
        #print("------------------------------------------\n",len(label_train),label_train)
        
        #print("------------------------------------------\n",len(label_test),label_test)

        #-----------------------------------------------------------Embedding
        feature_node_file = path + 'embed_node_features.csv'
        if not os.path.exists(feature_node_file):
            node_features = word_embedding_concept("../GoogleNews-vectors-negative300.bin.gz", indList)
            f = open(feature_node_file, 'w', encoding='utf-8', newline='')
            writer = csv.writer(f)
            writer.writerow(node_features[0])
            writer.writerows(node_features)
            f.close()
        else:
            node_features = pd.read_csv(feature_node_file, sep=',', encoding='utf-8')
        
        """
        feature_edge_file = path + 'embed_edge_features.csv'
        if not os.path.exists(feature_edge_file):
            edge_feature = word_embedding_concept("GoogleNews-vectors-negative300.bin.gz", objPropList)
            f = open(feature_edge_file, 'w', encoding='utf-8', newline='')
            writer = csv.writer(f)
            writer.writerow(edge_feature[0])
            writer.writerows(edge_feature)
            f.close()
        else:
            edge_feature = pd.read_csv(feature_edge_file, sep=',', encoding='utf-8')
        """
        #-----------------------------------------------------------make complete set of rule
        complete_set = []
        for x in edgeList:
            rule = str(indList[x[0]]) + " " + str(objPropList[x[2]]) + " " + str(indList[x[1]])
            complete_set.append(rule)

        #-----------------------------------------------------------make set of rule need to complement
        complement_set = []
        for x in edgeTest:
            rule = str(indList[x[0]]) + " " + str(objPropList[x[2]]) + " " + str(indList[x[1]])
            complement_set.append(rule)
        
        #make tensor for edge_feature
        #edge_feature = edge_feature.values.tolist()
        #edge_feat_list = []
        #for e in edge_type:
            #print("==========================================================================\n",e)
            #print("==========================================================================\n",edge_feature[e])
            #edge_feat_list.append(edge_feature[e])
        #edge_feat = torch.from_numpy(np.array(edge_feat_list)).float()

        #make tensor for edge_label
        #edge_label_graph = []
        #for i in range(len(edge_src)):
            #if [edge_src[i], edge_dst[i]] in edge_train_set:
                #edge_label_graph.append(train_label[edge_train_set.index([edge_src[i], edge_dst[i]])])
            #else:
                #edge_label_graph.append([0]*len(objPropList))
        #edge_label_graph = torch.from_numpy(np.array(edge_label_graph)).float()
        edge_label = torch.from_numpy(np.array(edge_label)).float()

        #norm edge
        _, inverse_index, count = np.unique((edge_dst, edge_type), axis=1, return_inverse=True,
                                        return_counts=True)
        degrees = count[inverse_index]  # c_{i,r} for each relation type
        edge_norm = np.ones(len(edge_dst), dtype=np.float32) / degrees.astype(np.float32)
        edge_type_train = torch.from_numpy(edge_type).unsqueeze(1) #for training
        edge_type = torch.from_numpy(edge_type).long() #for building graph
        edge_norm = torch.from_numpy(edge_norm).unsqueeze(1)
        #print("++++++++++++",edge_norm)
        #-----------------------------------------------------------make tensor for node_feature
        node_features = torch.from_numpy(np.array(node_features)).float()


        #--------------------------------------------------------make Graph
        # create multi-graph
        g = DGLGraph(multigraph=True)

        g_train = DGLGraph(multigraph=True)

        #thêm node vào graph
        g.add_nodes(num_node)

        g_train.add_nodes(num_node)

        #thêm cạnh vào graph-------------------src: node nguồn, dst: node đích
        g.add_edges(edge_src, edge_dst)

        g_train.add_edges(edge_src[train_mask], edge_dst[train_mask])

        #cập nhật type và chuẩn hóa các cạnh
        g.edata.update({'type': edge_type, 'norm': edge_norm})

        g_train.edata.update({'type': edge_type[train_mask], 'norm': edge_norm[train_mask]})

        # node_norm for apply_func
		#xử lý in_deg
        in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()

        in_deg_train = g_train.in_degrees(range(g_train.number_of_nodes())).float().numpy()

        #chuẩn hóa lại node (0,1)
        norm = 1.0 / in_deg
        norm[np.isinf(norm)] = 0
        node_norm = torch.from_numpy(norm)
        g.ndata.update({'norm': node_norm})


        norm_train = 1.0/in_deg_train
        norm_train[np.isinf(norm_train)] = 0
        node_norm_train = torch.from_numpy(norm_train)
        g_train.ndata.update({'norm': node_norm_train})

        #update feature and label
        g.ndata.update({'feature':node_features})

        g_train.ndata.update({'feature':node_features})

        #g.edata.update({'feature':edge_feat})
        g.edata.update({'label': edge_label})

        g_train.edata.update({'label': edge_label[train_mask]})

        #--------------------------------------------------------------------------get data
        edge_label = g.edata['label']
        edge_label_train = g_train.edata['label']

        in_feat = node_features.shape[1]
        num_classes = edge_label.shape[1]
        #print("---------------------------------------",len(edge_label[true_mask]))
        #----------------------------------------------------------------------create model base on BaseRGCN
        model = EdgeClassify( in_feat,
                              args.n_hidden,
                              num_classes,
                              num_rel,
                              num_bases=args.n_bases,
                              num_hidden_layers=args.n_layers - 2,
                              dropout=args.dropout,
                              use_cuda=use_cuda,
                              features=node_features)
        
        print("===============\nModel is created!")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)
        #loss_fcn = nn.CrossEntropyLoss(reduction='sum')
        #criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

        #make validation
        #val_idx = train_idx
        #val_labels = train_label
        print("number train:",len(edge_label[train_mask]))
        print("number test:",len(edge_label[test_mask]))
        print("number validation:",len(edge_label[val_mask]))

        model.train()
        print("start training...")

        #loss_function = torch.nn.MultiLabelSoftMarginLoss(reduction='mean')
        #loss_function = torch.nn.BCEWithLogitsLoss(reduction='sum')

        for epoch in range(args.n_epochs):
            print("======================== epoch", epoch,"========================")
            
            optimizer.zero_grad()
            pred = model.forward(g_train)
            
            #print("*************************************",F.hardsigmoid(pred[train_mask][0]))
            #print("*************************************",F.sigmoid(pred[train_mask][0]))
            #print("*************************************",pred[train_mask][0])
            #print("-------------------------------------",edge_label[train_mask][0])
            # similar loss
            #print(pred[train_mask].detach().numpy())
            #print(edge_type_train[train_mask].detach().numpy())

            #loss by edge_type
            loss = ((pred - edge_label_train) ** 2).sum()

            #loss = ((F.sigmoid(pred[train_mask]) - edge_label[train_mask]) ** 2).sum()

            #loss = ((F.softmax(pred[train_mask]) - edge_label[train_mask]) ** 2).sum()

            #loss = ((F.hardsigmoid(pred[train_mask]) - edge_label[train_mask]) ** 2).sum()
            
            #loss = F.binary_cross_entropy_with_logits(pred[train_mask], edge_label[train_mask], reduction='sum')

            #loss = F.cross_entropy(pred[train_mask], edge_label[train_mask])

            #loss = F.multilabel_margin_loss(pred[train_mask], edge_label[train_mask].long())

            #loss = loss_function(pred[train_mask], edge_label[train_mask])

            #loss = F.multilabel_margin_loss(pred[train_mask], edge_label[train_mask].long())

            #loss = F.multilabel_soft_margin_loss(F.sigmoid(pred[train_mask]), edge_label[train_mask].long())
            
            #backward
            print("loss: ",loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()

            #print label
            #print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\nAfter Change:",g.edata['label'].detach().numpy()[:10])
            #print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\nOutput label:",l[train_mask].detach().numpy()[:10])

            #evaluate
            #acc = evaluate(model, g, edge_label, val_mask)
            #print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f}". format(epoch, loss.item(), acc))
            print("---------------------------------------------------------")
        #evaluation
        model.eval()
        #l, score = model.forward(g)
        pred = model.forward(g)
        #print("==========================-------------------------------------score:\n",score)
        #pred = F.hardsigmoid(pred)
        #print("==============================predict: ",pred)
        #print("==============================predict[0]: ",pred[0])
        #print("==============================predict[10]: ",pred[10])
        #print("==============================predict[20]: ",pred[20])
        #print("==============================predict[30]: ",pred[30])
        #print("==============================predict[40]: ",pred[40])
        #print("==============================predict[50]: ",pred[50])
        #best_thred1, maxf11 = _select_threshold(edge_label[test_mask].detach().numpy(), pred[test_mask].detach().numpy())
        best_thred2, maxf12 = _select_threshold(edge_label[val_mask].detach().numpy(), pred[val_mask].detach().numpy())
        #best_thred3, maxf13 = _select_threshold(edge_label[train_mask].detach().numpy(), pred[train_mask].detach().numpy())
        #best_thred = max(best_thred1, best_thred2, best_thred3)
        best_thred = best_thred2
        print("===============\nbest_thred: ",best_thred)
        #print("===============\nmaxf1: ",maxf1)
        #labels_test_pre = np.zeros(test_label.shape)
        #labels_test_pre[np.where(l[test_mask].detach().numpy() > best_thred)] = 1
        labels_test_pre = np.zeros(edge_label[test_mask].shape)
        labels_test_pre[np.where(pred[test_mask].detach().numpy() > best_thred)] = 1
        
        labels_full_pre = np.zeros(edge_label.shape)
        labels_full_pre[np.where(pred.detach().numpy() > best_thred)] = 1
        #print(labels_test_pre)

        #find rules
        edge_src_test = edge_src[test_mask]
        edge_dst_test = edge_dst[test_mask]

        test_rules = []
        for r in range(len(test_label)):
            rules, _ = find_rule_edge(edge_src_test[r], edge_dst_test[r], test_label[r], indList, objPropList)
            test_rules += rules

        pred_rules = []
        for r in range(len(labels_test_pre)):
            rules, _ = find_rule_edge(edge_src_test[r], edge_dst_test[r], labels_test_pre[r], indList, objPropList)
            pred_rules += rules
        
        #find all rules
        all_new_rules = []
        full_triples = []
        for r in range(len(labels_full_pre)):
            rules, triples = find_rule_edge(edge_src[r], edge_dst[r], labels_full_pre[r], indList, objPropList)
            all_new_rules += rules
            full_triples += triples
        full_triples = np.array(full_triples, dtype=np.int)
        #add new rules in dataset
        with onto:
            for t in full_triples:
                s = indList[t[0]]
                p = objPropList[t[2]]
                o = indList[t[1]]
                if o not in list(p[s]):
                    p[s].append(o)
            #reasoner and save new dataset
            #sync_reasoner()
            #sync_reasoner_pellet(infer_property_values = True, infer_data_property_values = True, debug = 9)
            onto.save(file = newDataset, format = "rdfxml")
        #print("========================================Test:\n",test_rules)
        #print("========================================Pred:\n",pred_rules)

        #output
        output += "====================================================== Fold " + str(i) + ": \n"

        #Metrics
        test_precision, test_recall, test_f1 = metrics(set(test_rules), set(pred_rules))
        print("Test Precision: {:.4f} | Test Recall: {:.4f} | Test F1: {:.4f}".format(test_precision, test_recall, test_f1))
        output += "Test Precision: " + str(test_precision) + " | Test Recall: " + str(test_recall) + " | Test F1: " + str(test_f1) + "\n"

        #print the correct rules
        correct_rules = set(pred_rules).intersection(set(complement_set))
        print("==============================\nCorrect Prediction Rules ({}):".format(len(correct_rules)))
        output += "==============================\nCorrect Prediction Rules (" + str(len(correct_rules)) + "):\n"
        for r in correct_rules:
            print(r)
            output += r + "\n"
        #print the new rules
        new_rules = set(all_new_rules) - set(complete_set)
        #new_rules = set(pred_rules) - set(complete_set)
        print("==============================\nNew Rules ({})".format(len(new_rules)))
        output += "==============================\nNew Rules (" + str(len(new_rules)) + "):\n"
        for r in new_rules:
            #print(r)
            output += r + "\n"
        
        #make file output
        f = open(outputFile, 'w', encoding='utf-8')
        f.write(output)
        f.close()
        #del onto
        print("********************************************************************")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=64,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=0.01,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=3,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=150,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, default='biopax',
            help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")
    parser.add_argument("--sigma", type=float, default=0.1,
                        help="similar loss in coef")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="similar loss out coef")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    args.bfs_level = 0
    main(args)