from owlready2 import *
#from gensim.models import KeyedVectors
import numpy as np
import re
from num2words import num2words

#=========================================================
def sep_by_uppercase(str):
    #pattern = "[A-Z]"
    #result = ''.join(i for i in str if not i.isdigit())
    new_str = re.sub(r'[A-Z0-9]+', lambda x: " " + x.group(0), str)
    if str[0].islower():  # first character is lowercase
        return new_str
    else:
        return new_str[1:]

#=========================================================
def word_embedding_concept(embeding_file, classes):
    dim = 300
    embeddings = KeyedVectors.load_word2vec_format(embeding_file, binary=True)
    node_embeding = np.zeros((len(classes),dim))
    for concept in classes:

        #separer words
        words = concept.iri.strip().split('#')
        #print(words)
        if len(words) == 1:
            words = words[0].split('/')[-1][:-1]
            #print(words)
        else:
            words = words[1][:]
            #print(words)
        words = sep_by_uppercase(words)
        #print(words)

        #embedding
        if words in embeddings:
            node_embeding[classes.index(concept)] = embeddings[words] #test later
        else:
            if len(words) > 1:  # multi-words
                words = words.replace('-',' ').replace('_',' ').split(' ')
                vectors = np.zeros((len(words),dim))
                for w in words:
                    if w.isnumeric():
                        wordFromNumber = num2words(int(w))
                        wordFromNumber = wordFromNumber.split('-')
                        vectorNumber = np.zeros((len(wordFromNumber),dim))
                        for wnb in wordFromNumber:
                            if wnb in embeddings:
                                vectorNumber[wordFromNumber.index(wnb)] = embeddings[wnb]
                        vectors[words.index(w)] = np.mean(np.array(vectorNumber), axis=0)
                    else:
                        if w in embeddings:
                            vectors[words.index(w)] = embeddings[w]
                node_embeding[classes.index(concept)] = np.mean(np.array(vectors), axis=0)

    return node_embeding
#=========================================================
def word_embedding_ind(embeding_file, classes):
    dim = 300
    embeddings = KeyedVectors.load_word2vec_format(embeding_file, binary=True)
    node_embeding = np.zeros((len(classes),dim))

    for concept in classes:
        indOfClass = list(onto.search(type = concept))
        if len(indOfClass) > 0:
            ind_embedding = np.zeros((len(indOfClass),dim))
            for ind in indOfClass:

                #separer words
                words = ind.iri.strip().split('#')
                #print(words)
                if len(words) == 1:
                    words = words[0].split('/')[-1][:-1]
                    #print(words)
                else:
                    words = words[1][:]
                    #print(words)
                words = sep_by_uppercase(words)
                #print(words)
                
                #embedding
                if words in embeddings:
                    ind_embedding[indOfClass.index(ind)] = embeddings[words] #test later
                else:
                    if len(words) > 1:  # multi-words
                        words = words.replace('-',' ').replace('_',' ').split(' ')
                        vectors = np.zeros((len(words),dim))
                        for w in words:
                            if w.isnumeric():
                                wordFromNumber = num2words(int(w))
                                wordFromNumber = wordFromNumber.split('-')
                                vectorNumber = np.zeros((len(wordFromNumber),dim))
                                for wnb in wordFromNumber:
                                    if wnb in embeddings:
                                        vectorNumber[wordFromNumber.index(wnb)] = embeddings[wnb]
                                vectors[words.index(w)] = np.mean(np.array(vectorNumber), axis=0)
                            else:
                                if w in embeddings:
                                    vectors[words.index(w)] = embeddings[w]
                        ind_embedding[indOfClass.index(ind)] = np.mean(np.array(vectors), axis=0)
            node_embeding[classes.index(concept)] = np.mean(np.array(ind_embedding), axis=0)

    return node_embeding

#=========================================================
def word_embedding_concept_ind(embeding_file, classes):
    dim = 300
    embeddings = KeyedVectors.load_word2vec_format(embeding_file, binary=True)
    node_embeding = np.zeros((len(classes),dim))

    for concept in classes:
        #for concept
        concept_embedding = np.zeros(dim)
        #separer words
        words = concept.iri.strip().split('#')
        #print(words)
        if len(words) == 1:
            words = words[0].split('/')[-1][:-1]
            #print(words)
        else:
            words = words[1][:]
            #print(words)
        words = sep_by_uppercase(words)
        #print(words)

        #embedding
        if words in embeddings:
            concept_embedding = embeddings[words] #test later
        else:
            if len(words) > 1:  # multi-words
                words = words.replace('-',' ').replace('_',' ').split(' ')
                vectors = np.zeros((len(words),dim))
                for w in words:
                    if w.isnumeric():
                        wordFromNumber = num2words(int(w))
                        wordFromNumber = wordFromNumber.split('-')
                        vectorNumber = np.zeros((len(wordFromNumber),dim))
                        for wnb in wordFromNumber:
                            if wnb in embeddings:
                                vectorNumber[wordFromNumber.index(wnb)] = embeddings[wnb]
                        vectors[words.index(w)] = np.mean(np.array(vectorNumber), axis=0)
                    else:
                        if w in embeddings:
                            vectors[words.index(w)] = embeddings[w]
                concept_embedding = np.mean(np.array(vectors), axis=0)

        #for ind
        indOfClass = list(onto.search(type = concept))
        inds_embedding = np.zeros(dim)
        if len(indOfClass) > 0:
            ind_embedding = np.zeros((len(indOfClass),dim))
            for ind in indOfClass:

                #separer words
                iwords = ind.iri.strip().split('#')
                #print(iwords)
                if len(iwords) == 1:
                    iwords = iwords[0].split('/')[-1][:-1]
                    #print(iwords)
                else:
                    iwords = iwords[1][:]
                    #print(iwords)
                iwords = sep_by_uppercase(iwords)
                #print(iwords)
                
                #embedding
                if iwords in embeddings:
                    inds_embedding = embeddings[iwords] #test later
                else:
                    if len(iwords) > 1:  # multi-words
                        iwords = iwords.split(' ')
                        vectors = np.zeros((len(iwords),dim))
                        for w in iwords:
                            if w.isnumeric():
                                wordFromNumber = num2words(int(w))
                                wordFromNumber = wordFromNumber.split('-')
                                vectorNumber = np.zeros((len(wordFromNumber),dim))
                                for wnb in wordFromNumber:
                                    if wnb in embeddings:
                                        vectorNumber[wordFromNumber.index(wnb)] = embeddings[wnb]
                                vectors[iwords.index(w)] = np.mean(np.array(vectorNumber), axis=0)
                            else:
                                if w in embeddings:
                                    vectors[iwords.index(w)] = embeddings[w]
                        ind_embedding[indOfClass.index(ind)] = np.mean(np.array(vectors), axis=0)
            inds_embedding = np.mean(np.array(ind_embedding), axis=0)
        
        #Get embedding from concept and individuals
        if (np.count_nonzero(concept_embedding) == 0):
            node_embeding[classes.index(concept)] = inds_embedding
        elif (np.count_nonzero(inds_embedding) == 0):
            node_embeding[classes.index(concept)] = concept_embedding
        else:
            node_embeding[classes.index(concept)] = np.mean(np.array([concept_embedding,inds_embedding]), axis=0)

    return node_embeding
#=========================================================
def encodeMatrix(classes, individuals):
    oneHotMatrix = []
    for concept in classes:
        indOfClass = list(onto.search(type = concept))
        oneHot = [0]*len(individuals)
        for ind in indOfClass:
            oneHot[individuals.index(ind)] = 1
        oneHotMatrix.append(oneHot)
    return oneHotMatrix

#=========================================================
def relaInd(individuals):
    adjMatrix = []
    for s in individuals:
        oneHotTemp = [0]*len(individuals)
        oneHotTemp[individuals.index(s)] = 1
        for p in list(onto.object_properties()):
            for o in list(p[s]):
                if  o in individuals:
                    oneHotTemp[individuals.index(o)] = 1
                #print(str(s) + " " + str(p) + " " + str(o))
        adjMatrix.append(oneHotTemp)
    return adjMatrix

#=========================================================
def relaClass(classes):
    isA = []
    subClass = []
    sibling = []
    for concept in classes:
        father = concept.is_a
        isA.append(father[0])
        subClass.append(list(concept.subclasses()))
        sibling.append(list(father[0].subclasses()))
    return isA, subClass, sibling

#=========================================================
def subsumtion(classes):
    rule_base = []
    for concept in classes:
        for sub in list(concept.descendants()):
            rule_base.append([classes.index(sub),classes.index(concept)])
    return rule_base
#=========================================================
def edge_list1(objectProperties, individuals, onto):
    edgeList = []
    with onto:
        for s in individuals:
            for p in list(objectProperties):
                for o in list(p[s]):
                    if o in individuals:
                        edgeList.append([individuals.index(s),individuals.index(o),objectProperties.index(p)])
    return edgeList

#=========================================================
def label_concept(classes, individuals):
    global onto
    conceptDict = []
    
    for concept in classes:
        if (len(list(onto.search(type = concept))) > 0):
            conceptDict.append(concept)
    
    conceptLabel = np.zeros((len(individuals),len(conceptDict)))
    for concept in conceptDict:
        for ind in list(onto.search(type = concept)):
            conceptLabel[individuals.index(ind)][conceptDict.index(concept)] = 1

    return conceptDict, conceptLabel
#=========================================================

def label_edge(edge_list, objProps):
    edge_idx = []
    for e in edge_list:
        if [e[0],e[1]] not in edge_idx:
            edge_idx.append([e[0],e[1]])
    label = np.array([[0]*len(objProps)]*len(edge_idx))
    for e in edge_list:
        label[edge_idx.index([e[0],e[1]]),e[2]] = 1
    """
    for src in range(len(individuals)):
        labelTemp = []
        for dst in range((len(individuals))):
            labelTemp.append([src,dst,[0]*len(objProps)])
        #print(labelTemp[:10])
        for prob in range(len(objProps)):
            s = individuals[src]
            p = objProps[prob]
            for o in list(p[s]):
                if o in individuals:
                    #print(labelTemp[individuals.index(o)][2][prob])
                    labelTemp[individuals.index(o)][2][prob] = 1
                    #print(labelTemp[individuals.index(o)][2][prob])
        label += labelTemp
    """
    return edge_idx, label

#=========================================================
def make_template(edge_list):
    template_dict = []
    for ed in edge_list:
        if [ed[0], "?", ed[2]] not in template_dict:
            template_dict.append([ed[0], "?", ed[2]])
        if ["?", ed[1], ed[2]] not in template_dict:
             template_dict.append(["?", ed[1], ed[2]])
    with open('template_dict.txt', 'w') as f:
        for t in template_dict:
            f.write(str(t))
            f.write("\n")
    f.close()
    return template_dict

#=========================================================
def make_label(individuals, templateList, edges):
    label = np.array([[0]*len(templateList)]*len(individuals), dtype=np.int)
    for ed in edges:
        #make template
        template1 = []
        template1 += [ed[0],"?",ed[2]]
        template2 = []
        template2 += ["?",ed[1],ed[2]]
        #mark 1 for template
        if ((template1 in templateList) and (ed[1] in individuals)):
            label[individuals.index(ed[1]),templateList.index(template1)] = 1
        if ((template2 in templateList) and (ed[0] in individuals)):
            label[individuals.index(ed[0]),templateList.index(template2)] = 1
    #print(labelTemp)
    return label
#=========================================================

def find_rule(node, label1, templateDict, individuals, objectProperties):
    rules = []
    triples = []
    label = label1.tolist()
    for i in range(len(label)):
        if label[i] == 1:
            triple = []
            template = templateDict[i]
            #print("+_+_+_+_+_+_+_+_+_+_+_+_+_+\n",template)
            rule = ""
            #src
            if template[0] == "?":
                rule = rule + str(individuals[node])
                triple.append(node)
            else:
                rule = rule + str(individuals[template[0]])
                triple.append(template[0])
            
            #object properties
            rule = rule + " " + str(objectProperties[template[2]])
            
            #dst
            if template[1] == "?":
                rule = rule + " " + str(individuals[node])
                triple.append(node)
            else:
                rule = rule + " " + str(individuals[template[1]])
                triple.append(template[1])
            
            triple.append(template[2])

            rules.append(rule)
            triples.append(triple)
    return rules, triples

#=========================================================
def find_rule_edge(src, dst, label1, individuals, objectProperties):
    rules = []
    triple = []
    label = label1.tolist()
    for i in range(len(label)):
        if label[i] == 1:
            rule = ""
            #src
            rule = rule + str(individuals[src])
            #object properties
            rule = rule + " " + str(objectProperties[i])
            #dst
            rule = rule + " " + str(individuals[dst])

            rules.append(rule)
            triple.append([src,dst,i])
    return rules, triple
#=========================================================
#onto = get_ontology("biopax.owl")
#onto.load()
#sync_reasoner()
#indList = list(onto.individuals())
#classList = list(onto.classes())
#objPropList = list(onto.object_properties())
#isA, subClass, sibling = relaClass(classList)
#edgeList = edge_list1(objPropList, indList)
#conceptDict, conceptLabel = label_concept(classList, indList)
#edgeLabel = label_edge(indList, objPropList)
#encode = encodeMatrix(classList, indList)

#print("===============================\n",len(indList))
#print("===============================\n",len(conceptDict))
#print("===============================\n",conceptLabel)
#print("===============================\n",edgeLabel[1])
#print("===============================\n",len(edgeLabel))
"""
dem = 0
for x in edgeLabel:
    if (x[2].count(1)>0):
        dem += 1
        print("have edge")
print(dem)
print("===============================\n",len(edgeList))
"""
#for x in edgeLabel:
#    print("===============================\n",x)

#print("===============================\n",edgeList)
#print("===============================\n",list(onto.object_properties()))
#print("===============================\n",relaInd(indList))

#print("================================\n", classList[39])
#print("===============================\n",isA)
#print("==========================+====\n",subClass[39])
#print("==========================+====\n", onto.get_children_of(classList[39]))
#print("==========================+====\n", list(classList[39].subclasses()))
#print("===============================\n",sibling)
#print("===============================\n",subsumtion(classList))

#print(encode)
#print(indList[0].iri.strip().split('#'))
#print(sep_by_uppercase("bioSource43"))

#print(word_embedding_concept("GoogleNews-vectors-negative300.bin.gz", indList))

