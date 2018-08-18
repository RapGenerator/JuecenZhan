import urllib.parse
from bs4 import BeautifulSoup
import jieba
import jieba.analyse
import os


class Crawler:
    def __init__(self, word, wordnum):
        self.word = word
        self.wordnum = wordnum
        print(word)
        print(wordnum)


    def crawler_fun(self, url):
        try:
            response = urllib.request.urlopen(url, timeout=10)
            html_cont = response.read()
        except ValueError:
            print('current HTML is not exist')

        try:
            soup = BeautifulSoup(html_cont, 'html.parser', from_encoding='utf-8')
            summary_node = soup.find('div', class_="lemma-summary")
            res_data = summary_node.get_text()
        except Exception:
            print('data summary error!')
        return res_data

    def extract_words(self, text):
        fenci_text = jieba.cut(text)
        print(os.getcwd())
        stopwords = {}.fromkeys([line.rstrip() for line in open('expansion/stopwords.txt', mode='r', encoding='utf-8')])

        final = ""
        for word in fenci_text:
            if word not in stopwords:
                if (word != "。" and word != "，"):
                    final = final + " " + word
        ex_words = jieba.analyse.extract_tags(text, topK=self.wordnum, withWeight=True, allowPOS=())
        ex_words = [e[0] for e in ex_words]
        return ex_words

    def generate_list(self):
        name = urllib.parse.quote(self.word)
        name.encode('utf-8')
        root_url = 'http://baike.baidu.com/search/word?word=' + name
        data = self.crawler_fun(root_url)
        words = self.extract_words(data)
        return words



# if __name__ == '__main__':
#     word = '习近平'
#     name = urllib.parse.quote(word)
#     name.encode('utf-8')
#     root_url = 'http://baike.baidu.com/search/word?word=' + name
#
#     file_path = 'topic_words.txt'
#     data = crawler_fun(root_url)
#     words = extract_words(data)
#
#     if not os.path.exists(file_path):
#         f = open(file_path, 'w')
#     else:
#         f = open(file_path, 'a')
#     f.write(word + ' ' + str(words) + '\n')
#     print(word,words)