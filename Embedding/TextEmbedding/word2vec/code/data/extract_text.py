import logging
import os.path
import sys
from gensim.corpora import WikiCorpus
import time
begin = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
 
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s"% ' '.join(sys.argv))
 

    inp, outp = "enwiki-latest-pages-articles15.xml-p9244801p9518048.bz2", "wiki_data.txt"
    space = ' '
    i = 0
    output = open(outp,'w',encoding='utf-8')
    wiki = WikiCorpus(inp,lemmatize=False,dictionary={ })
    for text in wiki.get_texts():
        s = space.join(text)+"\n"
        output.write(s)
        i = i+1
        if(i% 10000 == 0):
            logger.info("Saved "+str(i) + " articles")
    output.close()
    logger.info("Finished Saved "+ str(i) +" articles")
 
    end = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("begin",begin)
    print("end  ",end)
