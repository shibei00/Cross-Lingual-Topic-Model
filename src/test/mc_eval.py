#-*- coding:utf-8 -*-
import time;
import numpy;
import os
import scipy;
import scipy.misc
import scipy.stats
import sys;

import numpy as np

"""
Evaluate the perplexity of generated bilingual topic model
"""
class MC_EVAL:
    def __init__(self, test_docs_en, test_docs_cn, model, model_name, corpus_name,):
        self.test_docs_en = test_docs_en
        self.test_docs_cn = test_docs_cn
        self.model = model
        self.model_name = model_name
        self.corpus_name = corpus_name
        self.unpack_model(model)
        self.parse_dict()
        self.parse_data();
        self._counter = 0
        
        # define the total number of document
        self._number_of_documents_en = len(self._word_idss_en)
        self._number_of_documents_cn = len(self._word_idss_cn)

        self._n_dk_en = np.zeros((self._number_of_documents_en, self._number_of_topics))
        self._n_dk_cn = np.zeros((self._number_of_documents_cn, self._number_of_topics))

        self._number_of_types_en = len(self._type_to_index_en)
        self._number_of_types_cn = len(self._type_to_index_cn)

        self._n_kv_en = np.zeros((self._number_of_topics, self._number_of_types_en))
        self._n_kv_cn = np.zeros((self._number_of_topics, self._number_of_types_cn))

        self._n_k_en = np.zeros(self._number_of_topics)
        self._n_k_cn = np.zeros(self._number_of_topics)
        # define the topic assignment for every word in every document, first indexed by doc_id id, then indexed by word word_pos
        self._k_dn_en = {};
        self._k_dn_cn = {}
        alpha_alpha = 1.0 / self._number_of_topics
        alpha_beta = 0.01

        # initialize a K-dimensional vector, valued at 1/K.
        self._alpha_alpha = numpy.zeros(self._number_of_topics) + alpha_alpha
        self._alpha_beta_en = numpy.zeros(self._number_of_types_en) + alpha_beta
        self._alpha_beta_cn = numpy.zeros(self._number_of_types_cn) + alpha_beta
        
        self.merge_topcis()
        
        self.random_initialize();

    def unpack_model(self, model):
        self._type_to_index_en = model['type_to_index_en']
        self._type_to_index_cn = model['type_to_index_cn']
        self._index_to_type_en = model['index_to_type_en']
        self._index_to_type_cn = model['index_to_type_cn']

        self._kv_en_ge = model['kv_en_ge']
        self._kv_cn_ge = model['kv_cn_ge']
        self._number_of_topics_ge = self._kv_en_ge.shape[0]
        self._number_of_topics = self._number_of_topics_ge
        print 'model unpacked successfully...'

    def merge_topcis(self,):
        self._kv_en_trans = np.dot(self._kv_en_ge, self.trans_matrix_en_cn)
        self._kv_cn_trans = np.dot(self._kv_cn_ge, self.trans_matrix_cn_en)

        self.normalize_by_row(self._kv_en_trans)
        self.normalize_by_row(self._kv_cn_trans)
        self._topic_word_en = self._kv_cn_trans
        self._topic_word_cn = self._kv_en_trans

        self._topic_word_en = (self._topic_word_en + self._alpha_beta_en) / (np.sum(self._topic_word_en, axis=1) + np.sum(self._alpha_beta_en))[:,np.newaxis]               
        self._topic_word_cn = (self._topic_word_cn + self._alpha_beta_cn) / (np.sum(self._topic_word_cn, axis=1) + np.sum(self._alpha_beta_cn))[:,np.newaxis]
        
    def parse_dict(self):
        file_dir = os.path.dirname(os.path.realpath(__file__))
        file_name = os.path.join(file_dir, 'ch_en_dict.dat')
        f = open(file_name)
        lines = f.readlines()
        f.close()
        self.trans_dict = {}
        self.trans_matrix_en_cn = np.zeros((len(self._type_to_index_en), len(self._type_to_index_cn)))
        self.trans_matrix_cn_en = np.zeros((len(self._type_to_index_cn), len(self._type_to_index_en)))
        for line in lines:
            terms = (line.strip()).split()
            if len(terms)==2:
                ch_term = terms[0]
                en_term = terms[1]
                if ch_term in self._type_to_index_cn and en_term in self._type_to_index_en:
                    if ch_term not in self.trans_dict:
                        self.trans_dict[ch_term] = set()
                    if en_term not in self.trans_dict:
                        self.trans_dict[en_term] = set()
                    self.trans_dict[ch_term].add(en_term)
                    self.trans_dict[en_term].add(ch_term)
                    en_term_index = self._type_to_index_en[en_term]
                    ch_term_index = self._type_to_index_cn[ch_term]
                    self.trans_matrix_en_cn[en_term_index][ch_term_index] = 1
                    self.trans_matrix_cn_en[ch_term_index][en_term_index] = 1
        self.normalize_by_row(self.trans_matrix_en_cn)
        self.normalize_by_row(self.trans_matrix_cn_en)
        print 'successfully parse dictionary...'

    def normalize_by_row(self, np_matrix):
        matrix_sum = np.sum(np_matrix, axis=1)
        index = matrix_sum > 0
        np_matrix[index,:] = np_matrix[index,:] / matrix_sum[index][:, np.newaxis]
        
    def random_initialize(self):
        # initialize the vocabulary, i.e. a list of distinct tokens.
        for doc_id in xrange(self._number_of_documents_en):
            self._k_dn_en[doc_id] = np.zeros(len(self._word_idss_en[doc_id]))
            for word_pos in xrange(len(self._word_idss_en[doc_id])):
                type_index = self._word_idss_en[doc_id][word_pos]
                topic_index = np.random.randint(self._number_of_topics)

                self._k_dn_en[doc_id][word_pos] = topic_index
                self._n_dk_en[doc_id, topic_index] += 1
                self._n_kv_en[topic_index, type_index] += 1
                self._n_k_en[topic_index] += 1

        for doc_id in xrange(self._number_of_documents_cn):
            self._k_dn_cn[doc_id] = np.zeros(len(self._word_idss_cn[doc_id]))
            for word_pos in xrange(len(self._word_idss_cn[doc_id])):
                type_index = self._word_idss_cn[doc_id][word_pos]
                topic_index = np.random.randint(self._number_of_topics)

                self._k_dn_cn[doc_id][word_pos] = topic_index
                self._n_dk_cn[doc_id, topic_index] += 1
                self._n_kv_cn[topic_index, type_index] += 1
                self._n_k_cn[topic_index] += 1
        
    def parse_data(self):
        doc_count = 0
        self._word_idss_en = []
        self._word_idss_cn = []
        
        for document_line in self.test_docs_en:
            word_ids = [];
            for token in document_line.split():
                if token not in self._type_to_index_en:
                    continue;
                
                type_id = self._type_to_index_en[token];
                word_ids.append(type_id);
            
            if len(word_ids)==0:
                sys.stderr.write("warning: document collapsed during parsing\n");
                continue;
            self._word_idss_en.append(word_ids);
            
            doc_count+=1
            if doc_count%10000==0:
                print "successfully parse %d english documents..." % doc_count;
        print "successfully parse %d english documents..." % (doc_count);

        doc_count = 0
        for document_line in self.test_docs_cn:
            word_ids = [];
            for token in document_line.split():
                if token not in self._type_to_index_cn:
                    continue;
                type_id = self._type_to_index_cn[token];
                word_ids.append(type_id);
            
            if len(word_ids)==0:
                sys.stderr.write("warning: document collapsed during parsing\n");
                continue;
            self._word_idss_cn.append(word_ids);
            
            doc_count+=1
            if doc_count%10000==0:
                print "successfully parse %d chinese documents..." % doc_count;
        print "successfully parse %d chinese documents..." % (doc_count);        
        
    def sample_document_en(self, doc_id, local_parameter_iteration=1):
        for iter in xrange(local_parameter_iteration):            
            for position in xrange(len(self._word_idss_en[doc_id])):
                assert position >= 0 and position < len(self._word_idss_en[doc_id])
                word_id = self._word_idss_en[doc_id][position]
                old_topic = self._k_dn_en[doc_id][position]
                self._n_dk_en[doc_id, old_topic] -= 1
                self._n_kv_en[old_topic, word_id] -= 1
                self._n_k_en[old_topic] -= 1

                log_probability = np.log(self._n_dk_en[doc_id, :] + self._alpha_alpha)
                log_probability += np.log(self._topic_word_en[:,word_id]);
                log_probability -= scipy.misc.logsumexp(log_probability)
                
                #sample a new topic out of a distribution according to log_probability
                temp_probability = np.exp(log_probability);
                temp_topic_probability = np.random.multinomial(1, temp_probability)[np.newaxis, :]
                new_topic = np.nonzero(temp_topic_probability == 1)[1][0];

                self._n_dk_en[doc_id, new_topic] += 1
                self._n_kv_en[new_topic, word_id] += 1
                self._n_k_en[new_topic] += 1
                self._k_dn_en[doc_id][position] = new_topic

    def sample_document_cn(self, doc_id, local_parameter_iteration=1):
        for iter in xrange(local_parameter_iteration):            
            for position in xrange(len(self._word_idss_cn[doc_id])):
                assert position >= 0 and position < len(self._word_idss_cn[doc_id])
                word_id = self._word_idss_cn[doc_id][position]
                old_topic = self._k_dn_cn[doc_id][position]
                self._n_dk_cn[doc_id, old_topic] -= 1
                self._n_kv_cn[old_topic, word_id] -= 1
                self._n_k_cn[old_topic] -= 1

                log_probability = np.log(self._n_dk_cn[doc_id, :] + self._alpha_alpha)
                log_probability += np.log(self._topic_word_cn[:,word_id])
                log_probability -= scipy.misc.logsumexp(log_probability)
                
                #sample a new topic out of a distribution according to log_probability
                temp_probability = np.exp(log_probability);
                temp_topic_probability = np.random.multinomial(1, temp_probability)[np.newaxis, :]
                new_topic = np.nonzero(temp_topic_probability == 1)[1][0];

                self._n_dk_cn[doc_id, new_topic] += 1
                self._n_kv_cn[new_topic, word_id] += 1
                self._n_k_cn[new_topic] += 1
                self._k_dn_cn[doc_id][position] = new_topic

    def perplexity(self,):
        log_likelihood_en = 0.0
        log_likelihood_cn = 0.0
        word_count_en = 0
        word_count_cn = 0
        perplexity = 0
        
        topic_word_probability_en = self._topic_word_en
        for doc_id in xrange(self._number_of_documents_en):
            doc_topic_probability_en = (self._n_dk_en[doc_id]  + self._alpha_alpha) / (np.sum(self._n_dk_en[doc_id]) + np.sum(self._alpha_alpha))
            word_count_en += len(self._word_idss_en[doc_id])
            for pos in xrange(len(self._word_idss_en[doc_id])):
                word_id = self._word_idss_en[doc_id][pos]
                log_likelihood_en += np.log(np.dot(doc_topic_probability_en, topic_word_probability_en[:, word_id]))

        topic_word_probability_cn = self._topic_word_cn
        for doc_id in xrange(self._number_of_documents_cn):
            doc_topic_probability_cn = (self._n_dk_cn[doc_id]  + self._alpha_alpha) / (np.sum(self._n_dk_cn[doc_id]) + np.sum(self._alpha_alpha))
            word_count_cn += len(self._word_idss_cn[doc_id])
            for pos in xrange(len(self._word_idss_cn[doc_id])):
                word_id = self._word_idss_cn[doc_id][pos]
                log_likelihood_cn += np.log(np.dot(doc_topic_probability_cn, topic_word_probability_cn[:, word_id]))

        perplexity_en = np.exp(-1.0 * log_likelihood_en / word_count_en)
        perplexity_cn = np.exp(-1.0 * log_likelihood_cn / word_count_cn)
        perplexity = np.exp(-1.0 * (log_likelihood_en + log_likelihood_cn) / (word_count_en + word_count_cn))
        return perplexity

    def learning(self):
        #sample the total corpus
        #for iter1 in xrange(number_of_iterations):
        self._counter += 1;
        processing_time = time.time();

        #sample every document
        count = 0
        for doc_id in xrange(self._number_of_documents_en):
            self.sample_document_en(doc_id)
            count += 1
            if count%1000==0:
                print "successfully sampled %d english documents" % (count,)

        count = 0
        for doc_id in xrange(self._number_of_documents_cn):
            self.sample_document_cn(doc_id)
            count += 1
            if count%1000==0:
                print "successfully sampled %d chinese documents" % (count,)

        processing_time = time.time() - processing_time;
        print "iteration %i finished in %d seconds with testing perplexity %g" % (self._counter, processing_time, self.perplexity())

    def test_model(self, topic_directory):
        file_name = self.model_name + '_' + str(self._number_of_topics_ge)  + '_'+self.corpus_name+'_'+'topics.txt'
        file_name = os.path.join(topic_directory, file_name)
        f = open(file_name, 'w')
        top = 20
        for topic_index in xrange(self._number_of_topics_ge):
            beta_probability = self._kv_en_ge[topic_index] * -1
            type_list = (np.argsort(beta_probability))[0:top]
            content = ' '.join([self._index_to_type_en[x] for x in type_list]) + '\n'
            f.write(content)
            

        for topic_index in xrange(self._number_of_topics_ge):
            beta_probability = self._kv_cn_ge[topic_index] * -1
            type_list = np.argsort(beta_probability)[0:top]
            content = ' '.join([self._index_to_type_cn[x] for x in type_list]) + '\n'
            f.write(content)
        f.close()

    def output_result(self, output_directory):
        perplexity_file = 'perplexity.txt'
        perplexity_file = os.path.join(output_directory, perplexity_file)
        f = open(perplexity_file, 'w')
        f.write('Testing perplexity:{0}'.format(str(self.perplexity())))
        print 'Testing perplexity:{0}'.format(str(self.perplexity()))
        f.close()

if __name__ == "__main__":
    print "not implemented"
