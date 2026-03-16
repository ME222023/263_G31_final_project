import gensim.downloader as api

wv = api.load("word2vec-google-news-300")  # downloads/caches if needed
wv.save_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)