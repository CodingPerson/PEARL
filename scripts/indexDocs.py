#!/usr/bin/env python
#coding=utf-8
# translate word into id in documents
import sys

w2id = {}

def indexFile(pt, res_pt):
    print('index file: '+str(pt))
    wf = open(res_pt, 'w')
    for l in open(pt):
        ws = l.strip().split()
        for w in ws:
            if w not in w2id:
                w2id[w] = len(w2id)
                
        wids = [w2id[w] for w in ws]  
        print(' '.join(map(str, wids)), file=wf)
         
    print('write file: '+str(res_pt))


def write_w2id(res_pt):
    print('write:'+str(res_pt))
    wf = open(res_pt, 'w')
    for w, wid in sorted(w2id.items(), key=lambda d:d[1]):
        print('%d\t%s' % (wid, w), file=wf)
        
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python %s <doc_pt> <dwid_pt> <voca_pt>' % sys.argv[0])
        print('\tdoc_pt    input docs to be indexed, each line is a doc with the format "word word ..."')
        print('\tdwid_pt   output docs after indexing, each line is a doc with the format "wordId wordId..."')
        print('\tvoca_pt   output vocabulary file, each line is a word with the format "wordId    word"')
        exit(1)
        
    doc_pt = sys.argv[1]
    dwid_pt = sys.argv[2]
    voca_pt = sys.argv[3]
    indexFile(doc_pt, dwid_pt)
    print('n(w)='+str(len(w2id)))
    write_w2id(voca_pt)
