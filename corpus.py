from collections import Iterable, defaultdict, Counter
import itertools

import numpy as np

class Corpus(object):
    """Wrapper class for corpus that is iterable

    Parameters
    ----------
    corpus : Iterable
        Corpus to be wrapped"""
    def __init__(self, corpus):
        if not isinstance(corpus, Iterable):
            raise TypeError("`corpus` must be Iterable.")        
        self.corpus = corpus

    def __iter__(self):
        """Implements iterator generator"""
        self.corpus, iter_copy = itertools.tee(self.corpus)
        for document in iter_copy:
            yield document

    def get_vector_space_repr(self):
        """Turn corpus into the vector space represenation

        Returns
        -------
        Vector space representation of each document as 2-tuple:
        (doc_id, term_count)"""
        term2id = self.get_term2id()
        for document in self:
            counter = Counter(document)
            vector_space = [(term2id[term], count) for term, count in
                counter.iteritems()]
            vector_space = sorted(vector_space, key=lambda x: x[0])

            yield vector_space 

    def get_term2id(self):
        """Build term-to-id dictionary

        Returns
        -------
        term2id : dict
            Term-to-id dictionary 
        """

        term2id = dict()
        index = 0
        for document in self:
            if not isinstance(document, Iterable):
                raise TypeError("`document` must be Iterable.")
            for term in document:
                if not isinstance(term, basestring):
                    raise TypeError("`term` must be str type.")

                if term not in term2id:
                    term2id[term] = index
                    index += 1
        self.term2id_ = term2id
        return term2id

    def tfidf_fit(self, normalize=True, tf_func=None, idf_func=None):
        """"Fit tfidf from wrapped corpus

        Parameters
        ----------
        normalize : bool
            If True, the tfidf vector of each document will be normalized
            to unit l2 norm

        tf_func : {None, function}
            User-defined function for transforming term-frequency (tf),
            default (tf_func is None) is identity function.
 
        idf_func : {None, function}
            User-defined function for transforming iverse-term-frequency (idf),
            default (idf_func is None) is `log2(n_document / idf)`
    
        Returns
        -------
        tfidf : list
            tfidf as list of list (docment) of 2-tuple (term_id, tfidf_value) 
        """
        if not isinstance(normalize, bool):
            raise TypeError("`normalize` must be bool.")
        if tf_func is not None and not callable(tf_func):
            raise TypeError("`tf_func` must be callable.")
        if idf_func is not None and not callable(idf_func):
            raise TypeError("`idf_func` must be callable.")

        vector_space, iter_copy = itertools.tee(self.get_vector_space_repr())
        df = defaultdict(int)
        n_doc = 0

        if tf_func is None:
            tf_func = lambda x: x

        if idf_func is None:
            idf_func = lambda x, n_doc: np.log2(n_doc / float(x))

        for doc in vector_space:
            n_doc += 1
            for tid, _ in doc:
                df[tid] += 1

        idf = {tid: idf_func(df[tid], n_doc) for tid in df.keys()}
        tfidf = [[(tid, idf[tid] * tf_func(count)) for tid, count in doc]
            for doc in iter_copy]

        if normalize:
            def norm_func(doc):
                vec = np.array(zip(*doc)[1])
                vec = vec / np.sqrt(vec.dot(vec))
                return zip(zip(*doc)[0], vec)
            
            tfidf = [norm_func(doc) for doc in tfidf]

        self.idf_ = idf
        self.tfidf_ = tfidf
        return tfidf
