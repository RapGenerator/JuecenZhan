# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import jieba

padToken, unknownToken, goToken, eosToken = 0, 1, 2, 3


class Batch:
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []


def load_and_cut_data(filepath):
    '''
    加载数据并分词
    :param filepath: 路径
    :return: data: 分词后的数据
    '''
    with open(filepath, 'r', encoding='UTF-8') as f:
        data = []
        lines = f.readlines()
        for line in lines:
            seg_list = jieba.cut(line.strip(), cut_all=False)
            cutted_line = [e for e in seg_list]
            data.append(cutted_line)
    return data


def create_dic_and_map(sources, targets):
    '''
    得到输入和输出的字符映射表
    :param sources:
           targets:
    :return: sources_data:
             targets_data:
             word_to_id: 字典，数字到数字的转换
             id_to_word: 字典，数字到汉字的转换
    '''
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']

    # 得到每次词语的使用频率
    # word_dic = {}
    # for line in (sources + targets):
    #     for character in line:
    #         word_dic[character] = word_dic.get(character, 0) + 1

    # 去掉使用频率为1的词
    # word_dic_new = []
    # for key, value in word_dic.items():
    #     if value > 1:
    #         word_dic_new.append(key)

    word_dic_new = list(set([character for line in (sources + targets) for character in line]))

    # 将字典中的汉字/英文单词映射为数字
    #special list is a list and word_dic_new is a dictionary. We enumerate both of them.
    #id_to_word is a dict: key is an interger id, value is a unique word
    id_to_word = {idx: word for idx, word in enumerate(special_words + word_dic_new)}
    word_to_id = {word: idx for idx, word in id_to_word.items()}

    # 将sources和targets中的汉字/英文单词映射为数字
    sources_data = [[word_to_id.get(character, word_to_id['<UNK>']) for character in line] for line in sources]
    targets_data = [[word_to_id.get(character, word_to_id['<UNK>']) for character in line] for line in targets]

    return sources_data, targets_data, word_to_id, id_to_word


def createBatch(sources, targets):
    #create a batch object
    batch = Batch()

    batch.encoder_inputs_length = [len(source) for source in sources] 
    #We need to increment target length by 1 because for each sentence in the list, EOS is addede to the end the sentence.
    batch.decoder_targets_length = [len(target) + 1 for target in targets]

    #The maximum length of all the input sentences.
    max_source_length = max(batch.encoder_inputs_length)
    #The maximum length of all the target sentences.
    max_target_length = max(batch.decoder_targets_length)

    for source in sources:
        # 将source进行反序并PAD
        # This is a trick for encoding process. 
        source = list(reversed(source)) #reverse the list
        pad = [padToken] * (max_source_length - len(source)) 
        batch.encoder_inputs.append(pad + source)

    for target in targets:
        # 将target进行PAD，并添加EOS符号
        pad = [padToken] * (max_target_length - len(target) - 1)
        eos = [eosToken] * 1
        batch.decoder_targets.append(target + eos + pad)

    return batch


def getBatches(sources_data, targets_data, batch_size):

    data_len = len(sources_data)

    #Each time we call genNextSamples, 我们拿出batch-size个句子
    def genNextSamples():
        #Every time we update i with i + batch-size
        for i in range(0, len(sources_data), batch_size): #（startIndex, endIndex(not included), step) 
            yield sources_data[i:min(i + batch_size, data_len)], targets_data[i:min(i + batch_size, data_len)]

    batches = []
    for sources, targets in genNextSamples():
        batch = createBatch(sources, targets)
        batches.append(batch)

    return batches


#This is a testing method by user
def sentence2enco(sentence, word2id):
    '''
    测试的时候将用户输入的句子转化为可以直接feed进模型的数据，现将句子转化成id，然后调用createBatch处理
    :param sentence: 用户输入的句子
    :param word2id: 单词与id之间的对应关系字典
    :return: 处理之后的数据，可直接feed进模型进行预测
    '''
    if sentence == '':
        return None
    # 分词
    seg_list = jieba.cut(sentence.strip(), cut_all=False)
    cutted_line = [e for e in seg_list]

    # 将每个单词转化为id
    wordIds = []
    for word in cutted_line:
        wordIds.append(word2id.get(word, unknownToken))
    print(wordIds)
    # 调用createBatch构造batch
    batch = createBatch([wordIds], [[]])
    return batch


# if __name__ == '__main__':
#     filepath = 'data/data.txt'
#     batch_size = 70
#     data = load_data(filepath)
#     processed_data, word_to_id, id_to_word = process_all_data(data)  # 根据词典映射
#     batches = getBatches(processed_data, batch_size)
#
#     temp = 0
#     for nexBatch in batches:
#         if temp == 0:
#             print(len(nexBatch.encoder_inputs))
#             print(len(nexBatch.encoder_inputs_length))
#             print(nexBatch.decoder_targets)
#             print(nexBatch.decoder_targets_length)
#         temp += 1
