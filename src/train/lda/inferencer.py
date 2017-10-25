#-*- coding:utf-8 -*-
import time
import numpy
import scipy
import nltk
import os

def compute_dirichlet_expectation(dirichlet_parameter):
    if (len(dirichlet_parameter.shape) == 1):
        return scipy.special.psi(dirichlet_parameter) - scipy.special.psi(numpy.sum(dirichlet_parameter))
    return scipy.special.psi(dirichlet_parameter) - scipy.special.psi(numpy.sum(dirichlet_parameter, 1))[:, numpy.newaxis]

class Inferencer():
    """
    """
    def __init__(self,
                 #local_parameter_iterations=50,
                 hyper_parameter_optimize_interval=10,
                 ):
        
        self._hyper_parameter_optimize_interval = hyper_parameter_optimize_interval;
        assert(self._hyper_parameter_optimize_interval>0);
        
        #self._local_parameter_iterations = local_parameter_iterations
        #assert(self._local_maximum_iteration>0)        

    """
    """
    def _initialize(self, voc_en, voc_cn, number_of_topics_ge, alpha_alpha, alpha_beta, lam):
        self.parse_vocabulary(voc_en, voc_cn)
        self.parse_dictionary()
        # initialize the size of the vocabulary, i.e. total number of distinct tokens.
        self._number_of_types_en = len(self._type_to_index_en)
        self._number_of_types_cn = len(self._type_to_index_cn)
        
        self._counter = 0;
        
        # initialize the total number of topics.
        self._number_of_topics_ge = number_of_topics_ge
        self._number_of_topics = self._number_of_topics_ge
        
        # initialize a K-dimensional vector, valued at 1/K.
        self._alpha_alpha = numpy.zeros(self._number_of_topics) + alpha_alpha
        self._alpha_beta_en = numpy.zeros(self._number_of_types_en) + alpha_beta
        self._alpha_beta_cn = numpy.zeros(self._number_of_types_cn) + alpha_beta
        self.lam = lam

    def parse_dictionary(self):
        file_dir = os.path.dirname(os.path.realpath(__file__))
        file_name = os.path.join(file_dir, 'ch_en_dict.dat')
        f = open(file_name)
        lines = f.readlines()
        f.close()
        self.trans_dict = {}
        self.trans_list = []
        for line in lines:
            terms = (line.strip()).split()
            if len(terms)==2:
                ch_term = terms[0]
                en_term = terms[1]
                if ch_term in self._type_to_index_cn and en_term in self._type_to_index_en and ch_term in self._type_to_index and en_term in self._type_to_index:
                    if ch_term not in self.trans_dict:
                        self.trans_dict[ch_term] = set()
                    if en_term not in self.trans_dict:
                        self.trans_dict[en_term] = set()
                    self.trans_dict[ch_term].add(en_term)
                    self.trans_dict[en_term].add(ch_term)
                    self.trans_list.append((self._type_to_index[en_term], self._type_to_index[ch_term]))
        print 'successfully parse dictionary...'
    
    def parse_vocabulary(self, voc_en, voc_cn):
        self._type_to_index_en = {};
        self._index_to_type_en = {};
        self._type_to_index = {}
        self._index_to_type = {}
        voc = set(list(voc_en) + list(voc_cn))
        for word in set(voc_en):
            self._index_to_type_en[len(self._index_to_type_en)] = word;
            self._type_to_index_en[word] = len(self._type_to_index_en);
        self._voc_en = self._type_to_index_en.keys();

        self._type_to_index_cn = {};
        self._index_to_type_cn = {};
        for word in set(voc_cn):
            self._index_to_type_cn[len(self._index_to_type_cn)] = word;
            self._type_to_index_cn[word] = len(self._type_to_index_cn);
        self._voc_cn = self._type_to_index_cn.keys();

        for word in set(voc):
            self._index_to_type[len(self._index_to_type)] = word
            self._type_to_index[word] = len(self._type_to_index)
            
    def parse_data(self):
        raise NotImplementedError;

    """
    """
    def learning(self):
        raise NotImplementedError;
    
    """
    """
    def inference(self):
        raise NotImplementedError;

    def export_beta(self, exp_beta_path, top_display=-1):
        raise NotImplementedError;

    def export_topic_word(self):
        raise NotImplementedError
        
if __name__ == "__main__":
    raise NotImplementedError;
