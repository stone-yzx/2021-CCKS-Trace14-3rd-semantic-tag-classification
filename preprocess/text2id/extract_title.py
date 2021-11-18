# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 20:22:21 2021

@author: jerrycen
"""

import jieba
import string
import re
import os


class ExtractorTitle():

	def __init__(self,model_path):
		jieba.load_userdict(os.path.join(model_path, 'word_1_column.txt'))
		self.word_dict = self.get_word_id_dict(os.path.join(model_path, 'tencent_ailab_50w_200_emb_vocab.txt'))



	def delect_punctuation(self,str_text):
		str_text = ''.join(c for c in str_text if c not in string.punctuation)
		post_text=re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", str_text)
		return post_text

	def get_word_id_dict(self,in_file):
		word_id_dict={}
		with open(in_file,'r',encoding='utf-8')as fr:
			for line in fr:
				line_split=line.rstrip('\n').split('\t')
				if len(line_split)!=2:
					continue
				token=line_split[0]
				id_index=line_split[1]
				if token not  in word_id_dict:
					word_id_dict[token]=id_index

		return word_id_dict




	def pre_process_title(self,title):

		index = title.find('@{uid')
		if index != -1:
			title = title[:index]
		post_title = self.delect_punctuation(title)
		post_title = "".join([character.lower() for character in post_title])

		return post_title


	def cut_title_and_map_to_id_cut_twice(self,post_title):
		"""
		新的切词方法，当一个词被映射为1时，将其按全量模式切开，若还有匹配不上的词，则将其按字切开
		"""


		post_title_id = []

		post_title_token = jieba.cut(post_title, cut_all=False)
		for token in post_title_token:
			if token in self.word_dict:
				post_title_id.append(self.word_dict[token])
			else:
				token_cut_all = jieba.cut(token, cut_all=True)
				for sub_token in token_cut_all:
					if sub_token in self.word_dict:
						post_title_id.append(self.word_dict[sub_token])
					else:
						for character in sub_token:
							post_title_id.append(self.word_dict.get(character, '1'))

		post_title_str = ";".join(post_title_id)

		return post_title_str

	def extract_title(self,raw_title):
		title_id = self.cut_title_and_map_to_id_cut_twice(self.pre_process_title(raw_title))
		return title_id


def main():
	pass
if __name__ == '__main__':
    main()