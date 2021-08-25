import dgl.function as fn
from dgl import DGLGraph
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import RGCNBasisLayer as RGCNLayer
from model import BaseRGCN
from utils import _select_threshold, metrics, find_same_etype
from itertools import combinations
import itertools as it
import warnings
import os
from owlready2 import *
from gensim.models import KeyedVectors
import re
from num2words import num2words
from dataProcessing import *
from sklearn.model_selection import train_test_split
import csv
import pandas as pd
from random import randint

warnings.filterwarnings('ignore')


class NodeClassify(BaseRGCN):
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


# similar loss
def similar_loss(g, node_idx, logits, direction='in'):
    all_loss = 0
    for i in node_idx:
        if direction == 'in':
            u, _, eid = g.in_edges(i, 'all') #return [src, dst, etype]
        else:
            _, u, eid = g.out_edges(i, 'all')
        etype = g.edata['type'][eid]
        loss = 0
        for idx in find_same_etype(etype):
            sm_nodes = u[idx[1]]
            cnt = 0
            l = 0
            for n1, n2 in combinations(sm_nodes, 2):
                cnt += 1
                l += torch.norm(logits[n1] - logits[n2], 1)
            loss = l / cnt
        all_loss += loss
    #return all_loss / len(node_idx)
    return all_loss


def main(args):
    fold = 1

	#lấy sigma, beta từ commande line
    sigma = args.sigma
    beta = args.beta
	
    #manager input and output folder
    path = args.dataset + "/"
    pathData = "../rdf_xml_dataset/"
    dataset = pathData + args.dataset + ".owl"
    if not os.path.exists(path):
        os.mkdir(path)
    ontology = get_ontology(dataset)

	#lặp fold
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
		#num_node: số lượng node được mã hóa
		#edge_list: danh sách các cạnh của mạng
		#edge_src: danh sách nút nguồn
		#edge_dst: danh sách nút đích
		#edge_type: danh sách quan hệ giữa src và dst
		#edge_norm: chuẩn hóa của cạnh
		#num_rel: tổng số loại quan hệ
		#train_idx: id các node được xét trong train
		#train_label: label cho các nút được xét trong train
		#test_label: label cho các nút được xét trong test
		#node_features: embedding của node
        
        print("number of relations: ", len(edgeList))
        
        #split train/test set
        rand_split = randint(1,99)
        edgeTrainIdx, edgeTestIdx, edge_list, edgeTest = train_test_split(edgeIdx, edgeList, test_size=0.1, random_state=rand_split)
        #edgeTrainIdx = edgeIdx
        #edgeTestIdx = edgeIdx
        #edge_list = edgeList
        #edgeTest = edgeList

        #make template
        template_dict = make_template(edge_list)
        templateIdx = list(range(0,len(template_dict)))


        edge_src, edge_dst, edge_type = edge_list.transpose()
        num_rel = len(objPropList)

        #make self connect
        #edge_src = np.concatenate([edge_src,np.array(node_idx)])
        #edge_dst = np.concatenate([edge_dst,np.array(node_idx)])
        #edge_type = np.concatenate([edge_type, np.array([len(objPropList)]*num_node)])
        #num_rel += 1

        #make norm
        _, inverse_index, count = np.unique((edge_dst, edge_type), axis=1, return_inverse=True,
                                        return_counts=True)
        degrees = count[inverse_index]  # c_{i,r} for each relation type
        edge_norm = np.ones(len(edge_dst), dtype=np.float32) / degrees.astype(np.float32)

        #make train_idx
        train_idx = []
        for t in templateIdx:
            src = template_dict[t][0]
            dst = template_dict[t][1]
            if ((src not in train_idx) and (src != "?")):
                train_idx.append(src)
            if ((dst not in train_idx) and (dst != "?")):
                train_idx.append(dst)
        train_idx.sort()
        #print("------------------------------------------\n",len(train_idx),num_node)

        #make test_idx
        test_idx = []
        for e in edgeTest:
            src = e[0]
            dst = e[1]
            if src not in test_idx:
                test_idx.append(src)
            if dst not in test_idx:
                test_idx.append(dst)
        test_idx.sort()
        #print("------------------------------------------\n",test_idx)

        #make label
        train_label = np.array(make_label(train_idx, template_dict, edge_list), dtype=np.int)
        test_label = np.array(make_label(test_idx, template_dict, edgeList), dtype=np.int)
        #print("------------------------------------------\n",np.sum(test_label))
        

        feature_file = path + 'embed_features.csv'
        if not os.path.exists(feature_file):
            node_features = word_embedding_concept("../GoogleNews-vectors-negative300.bin.gz", indList)
            f = open(feature_file, 'w', encoding='utf-8', newline='')
            writer = csv.writer(f)
            writer.writerow(node_features[0])
            writer.writerows(node_features)
            f.close()
        else:
            node_features = pd.read_csv(feature_file, sep=',', encoding='utf-8')
        
        #make complete set of rule
        complete_set = []
        for x in edgeList:
            rule = str(indList[x[0]]) + " " + str(objPropList[x[2]]) + " " + str(indList[x[1]])
            complete_set.append(rule)

        #make set of rule need to complement
        complement_set = []
        for x in edgeTest:
            rule = str(indList[x[0]]) + " " + str(objPropList[x[2]]) + " " + str(indList[x[1]])
            complement_set.append(rule)

        #make set of rule need to complement
        train_set = []
        for x in edge_list:
            rule = str(indList[x[0]]) + " " + str(objPropList[x[2]]) + " " + str(indList[x[1]])
            train_set.append(rule)
        
        #print("===============================\n",nodeFeatures[:20])
        
		#---------------------------------index
        train_idx = list(train_idx)

        edge_type = torch.from_numpy(edge_type).long()
        edge_norm = torch.from_numpy(edge_norm).unsqueeze(1)

		#chuyển node_features sang dạng tensor 
        #print(node_features)
        node_features = torch.from_numpy(np.array(node_features)).float()
        #print(node_features)
        # check cuda -> nếu có thì truyền các edge vào 
        use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(args.gpu)
            edge_type = edge_type.cuda()
            edge_norm = edge_norm.cuda()
            node_features = node_features.cuda()

		####################---------------tạo multi graph
        # create multi-graph
        g = DGLGraph(multigraph=True)
		#thêm node vào graph
        g.add_nodes(num_node)
		#thêm cạnh vào graph-------------------src: node nguồn, dst: node đích
        g.add_edges(edge_src, edge_dst)

		#cập nhật type và chuẩn hóa các cạnh
        g.edata.update({'type': edge_type, 'norm': edge_norm})
		#-----------------------------------số đường liên kết của dst
        # node_norm for apply_func
		#xử lý in_deg
        in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
        #chuẩn hóa lại node (0,1)
        norm = 1.0 / in_deg
        norm[np.isinf(norm)] = 0
        node_norm = torch.from_numpy(norm)
		
		#nếu sử dụng cuda: update lại node_norm
        if use_cuda:
            node_norm = node_norm.cuda()
        g.ndata.update({'norm': node_norm})
		
		#xử lý in_feat
        #lấy shape input
        in_feat = node_features.shape[1]
        #print("******************************************\n",in_feat)
		#số lượng class (phân loại multi label)
        num_classes = train_label.shape[1]
        #print("******************************************\n",num_classes)
		
        #tensor of train and test set:
        train_label = torch.from_numpy(np.array(train_label)).float()
        test_label = torch.from_numpy(np.array(test_label)).float()
        
        # create model base on BaseRGCN
        model = NodeClassify(in_feat,
                              args.n_hidden,
                              num_classes,
                              num_rel,
                              num_bases=args.n_bases,
                              num_hidden_layers=args.n_layers - 2,
                              dropout=args.dropout,
                              use_cuda=use_cuda,
                              features=node_features)
        print("===============\nModel is created!")

        if use_cuda:
            model.cuda()
            train_label.cuda()
            test_label.cuda()

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
        #criterion = torch.nn.MultiLabelMarginLoss(reduction='sum')

        #if args.validation:
        #    train_idx, val_idx, train_label, val_labels = train_test_split(train_idx, train_label, test_size=0.10, random_state=rand_split)
        #else:
        #    val_idx = train_idx
        #    val_labels = train_label
        #val_idx = train_idx
        #val_labels = train_label
        #train_idx, val_idx, train_label, val_labels = train_test_split(train_idx, train_label, test_size=0.10, random_state=rand_split)
        #print("==============================Train label:\n",train_label)
        #print("==============================Train label size: \n",train_label.size())
        #print("==============================Test size: \n",test_label.size())

        print("number train:",len(train_label))
        print("number test:",len(test_label))
        #print("number validation:",len(val_labels))

        model.train()
        print("start training...")

        #print("Training...")

        #BaseRGCN model is created
        #====================================================================================================================


        for epoch in range(args.n_epochs):
            print("======================== epoch", epoch,"========================")

            optimizer.zero_grad()
            logits = model.forward(g)

            # similar loss
            sim_loss_in = similar_loss(g, train_idx, logits, 'in')
            sim_loss_out = similar_loss(g, train_idx, logits, 'out')

            #classify_loss
            #classify_loss = criterion(logits[train_idx], train_label)
            #classify_loss = criterion(logits[train_idx], train_label.long())  # for multi-label classification
            classify_loss = ((logits[train_idx] - train_label) ** 2).sum()

            #backward
            loss = classify_loss + sigma * sim_loss_in + beta * sim_loss_out
            print("loss: ",loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()
            print("--------------------------------------------------------")
        model.eval()
        logits = model.forward(g)
        #logits = F.sigmoid(logits)
        best_thred, maxf1 = _select_threshold(test_label, logits[test_idx].detach().numpy())
        print("===============\nbest_thred: ",best_thred)
        print("===============\nmaxf1: ",maxf1)
        labels_test_pre = np.zeros(test_label.shape)
        labels_test_pre[np.where(logits[test_idx].detach().numpy() > best_thred)] = 1

        labels_full_pre = np.zeros(logits.shape)
        labels_full_pre[np.where(logits.detach().numpy() > best_thred)] = 1

        #print("==============================testsize: ", labels_test_pre.shape)
        #print("==============================Test label: \n",test_label)
        #print("==============================Test label size: ",test_label.size())
        #print("==============================Node: \n", test_idx[2])
        
        """
        x = 27
        rule_test_2 = find_rule(test_idx[x], test_label[x].numpy(), template_dict, indList, objPropList)
        rule_pred_2 = find_rule(test_idx[x], labels_test_pre[x], template_dict, indList, objPropList)
        print("==============================Node_idx: \n", test_idx[x])
        print("==============================Test Label[x]: \n", rule_test_2)
        print("==============================Predict label[x] predict: \n", rule_pred_2)
        print("==============================number of rules: \n",len(rule_test_2), len(rule_pred_2))
        print("==============================same rule: \n", set(rule_test_2).intersection(set(rule_pred_2)))

        rule_test_0 = find_rule(test_idx[0], test_label[0].numpy(), template_dict, indList, objPropList)
        rule_pred_0 = find_rule(test_idx[0], labels_test_pre[0], template_dict, indList, objPropList)
        print("==============================Node_idx: \n", test_idx[0])
        print("==============================Test Label[0]: \n", rule_test_0)
        print("==============================Predict label[0] predict: \n", rule_pred_0)
        print("==============================number of rules: \n",len(rule_test_0), len(rule_pred_0))
        print("==============================same rule: \n", set(rule_test_0).intersection(set(rule_pred_0)))
        #print("==============================Number of edges: \n",np.sum(test_label))
        #print("==============================Number of edges predict: \n",np.sum(labels_test_pre))
        #print("==============================Predict label size: \n",labels_test_pre.shape)
        """
        test_rules = []
        for r in range(len(test_label)):
            rules, _ = find_rule(test_idx[r], test_label[r].numpy(), template_dict, indList, objPropList)
            test_rules += rules

        pred_rules = []
        for r in range(len(labels_test_pre)):
            rules, _ = find_rule(test_idx[r], labels_test_pre[r], template_dict, indList, objPropList)
            pred_rules += rules
        
        #find all rules
        all_new_rules = []
        full_triples = []
        #print("************************************",labels_full_pre[0])
        for r in range(len(labels_full_pre)):
            rules, triples = find_rule(node_idx[r], labels_full_pre[r], template_dict, indList, objPropList)
            all_new_rules += rules
            full_triples += triples
        full_triples = np.array(full_triples, dtype=np.int)
        #add new rules in dataset
        #print(full_triples[:10].tolist())
        #print(edgeList[:10].tolist())
        for t in full_triples:
            s = indList[t[0]]
            p = objPropList[t[2]]
            o = indList[t[1]]
            if o not in list(p[s]):
                p[s].append(o)
        
        #output
        output += "====================================================== Fold " + str(i) + ": \n"

        #Metrics
        test_precision, test_recall, test_f1 = metrics(set(test_rules), set(pred_rules))
        #test_precision, test_recall, test_f1 = metrics(set(train_set), set(pred_rules).intersection(set(train_set)))
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
        print("==============================\nNew Rules ({})".format(len(new_rules)))
        output += "==============================\nNew Rules (" + str(len(new_rules)) + "):\n"
        for r in new_rules:
            #print(r)
            output += r + "\n"

        #reasoner and save new dataset
        #sync_reasoner()
        onto.save(file = newDataset, format = "rdfxml")
    
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