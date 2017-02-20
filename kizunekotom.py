# coding: utf-8
import os
import math
import glob
import itertools
import re
import MeCab
import pickle
import copy
def main():
  m = MeCab.Tagger('-Owakati')
  for meta in ['train', 'test']:
    f = open('./tasks_1-20_v1-2/ja-10k/ja.single-supporting-fact_{}.txt'.format(meta))
    counter = 0
    contents_holder = []
    contents_block = []
    for line in f.read().split('\n'):
      if '[1m' in line and '->' not in line and line[0] != ' ' and '(' not in line:
        line = line.replace('Sandra', 'サンドラ')
        line = line.replace('バスルーム', 'トイレ')
        line = line.replace('ベッドルーム', '寝室')
        line = line.replace('キッチン', '台所')
        text = re.sub(r'.\[22m', '', line) 
        text = re.sub(r'.\[1m', '', text) 
        ents = m.parse(text).strip().split()
        try:
          header = int(ents[0])
          #print(ents)
          if counter < header:
            counter = header
            contents_block.append( list(filter(lambda x:re.match(r'\d', x) == None, ents)) )
          else:
            counter = 1
            contents_holder.append( copy.copy(contents_block) )
            contents_block = [ \
	      list(filter(lambda x:re.match(r'\d', x) == None,ents) ) \
	    ]
        except:
          counter += 1
          contents_block.append(ents)
    dataset = []
    """翻訳失敗で["0"]が入ってしまうデータがあるので取り除く"""
    for i, block in enumerate(contents_holder):
      if [0] in block: continue
      residencial = []
      for bi, lines in enumerate(block):
        is_shitsumon = True if '？' in lines else False
        q = []
        a = []
        if is_shitsumon:
          q = lines[0:lines.index('？')+1]
          a = lines[lines.index('？')+1:]
        #print(i, is_shitsumon, lines, q, a)
        if is_shitsumon:
          dataset.append( (sum(filter(lambda x: '？' not in x, block[:bi]), []), q, a[0]) )

    open('./tasks_1-20_v1-2/ja-10k/ja.single-supporting-fact_{}.pkl'.format(meta), 'wb').write(pickle.dumps(dataset))
    for data in dataset:
      print(meta, data)
if __name__ == '__main__':
  main()
