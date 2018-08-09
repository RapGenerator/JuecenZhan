from collections import OrderedDict
from operator import itemgetter
import jieba
import jieba.posseg
import copy


if __name__ == '__main__':
	data = load_and_cut_data('df_all.txt')
	mydic = initializeDic(data)
	getMapping(mydic)

	topK_dic = getTopK(mydic, 10)
	createVocabTable(mydic)
	VocabDictionary = createVocabDictionary()
	createCoOccurrenceTable(VocabDictionary, topK_dic, 10)

part_of_speech = ['n', 'nr', 'ns', 'vr', 'vn', 'a', 'ad']

#read the source file and use Jieba package to split Chinese sentences and label their part of speech
#Data is a two-dimensional array and each element is a tuple: (word, part of speech).
def load_and_cut_data(filepath):
    with open(filepath, 'r', encoding='UTF-8') as f:
        data = []
        lines = f.readlines()
        for line in lines:
            #seg_list = jieba.cut(line.strip(), cut_all=False)
            seg = jieba.posseg.cut(line)
            cutted_line = []
            for i in seg:
                if ((i.word != ' ') & (i.word != '\n')):
                    cutted_line.append((i.word, i.flag))
            data.append(cutted_line)
    return data

#Initialize a large dictionary. Key is the all the words in the source, value is an empty dictionary
def initializeDic(data):
    dic = {}
    for line in data:
        for word in line:
            print(word[0])
            if word[0] in dic:
                continue
            else:
                dic[word[0]] = {}
    return dic

#Get the occurrence frequency of a word's neighbors.
def getMapping(dic, part_of_speech):
    for line in data:
        for i in range(len(line)):
            for j in range(len(line)):
                if (line[i][0] == line[j][0]):
                    continue
                if (line[j][1] not in part_of_speech):
                    continue
                else:
                    if line[j][0] in dic[line[i][0]]:
                        dic[line[i][0]][line[j][0]] = dic[line[i][0]][line[j][0]] + 1
                    else:
                        dic[line[i][0]][line[j][0]] = 1

#Sort the frequency of the dictionary inside the large dictionary 
#and return the top n frequent pairs for each key in the large dictionary 
def getTopK(mydic, n):
    import copy
    dic_copy = copy.deepcopy(mydic)
    for mykey in dic_copy:
        value_len = len(dic_copy[mykey])
        if (value_len < n):
            continue
        dic_copy[mykey] = {k: dic_copy[mykey][k] for k in list(dic_copy[mykey])[:n]}
    return dic_copy


#write the id->word pairs to csv
def createVocabTable(mydic):
    word_set = set()
    for key in mydic:
        word_set.add(key)
        for word in mydic[key]:
            word_set.add(word)
    my_list = []
    ##convert a set to list
    for element in word_set:
        my_list.append(element)
    ##should use list(word_set) 
    vocabTable = pd.DataFrame(my_list, columns = ['word'])
    vocabTable.index.rename('id',inplace = True)
    vocabTable.to_csv('vocabTable.csv')

#return the id->word dictionary
def createVocabDictionary():
    vocabTable = pd.read_csv('vocabTable.csv')
    VocabDictionary = {}
    row = 0
    while (row < len(vocabTable)):
        VocabDictionary[vocabTable.iloc[row, 1]] = vocabTable.iloc[row, 0]
        row = row + 1
    return VocabDictionary


#create the Co-occurrence table and write it to csv
def createCoOccurrenceTable(VocabDictionary, topK_dic, k):
    rows = []
    for key in topK_dic:
        word1_id = VocabDictionary[key]
        for innerKey in topK_dic[key]:
            word2_id = VocabDictionary[innerKey]
            count = topK_dic[key][innerKey]
            row = [word1_id, word2_id, count]
            rows.append(row)
    cooccurrence_df = pd.DataFrame(data = rows, columns=('word1_id', 'word2_id', 'count'))
    filepath = "co-occurrence_" + str(k) + ".csv"
    cooccurrence_df.to_csv(filepath)



