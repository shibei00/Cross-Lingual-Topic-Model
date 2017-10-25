#-*- coding:utf-8 -*-
import time;
import numpy as np;
import scipy;
import sys;
from inferencer import compute_dirichlet_expectation
from inferencer import Inferencer;

"""
This is a python implementation of bilingual topic model on the unaligned corpus, based on collapsed Gibbs sampling.

References:
@inproceedings{
  author    = {Bei Shi and
               Wai Lam and
               Lidong Bing and
               Yinqing Xu},
  title     = {Detecting Common Discussion Topics Across Culture From News Reader
               Comments},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational
               Linguistics, {ACL} 2016, August 7-12, 2016, Berlin, Germany, Volume
               1: Long Papers},
  year      = {2016},
}
"""

class CrossLDA(Inferencer):
    def __init__(self,
                 hyper_parameter_optimize_interval=10,
                 symmetric_alpha_alpha=True,
                 symmetric_alpha_beta=True,
                 #local_parameter_iteration=1,
                 ):
        Inferencer.__init__(self, hyper_parameter_optimize_interval);

        self._symmetric_alpha_alpha=symmetric_alpha_alpha
        self._symmetric_alpha_beta=symmetric_alpha_beta

                
    """
    @param num_topics: desired number of topics
    @param data: a dict data type, indexed by document id, value is a list of words in that document, not necessarily be unique
    """
    def _initialize(self, corpus_en, voc_en, corpus_cn, voc_cn,  number_of_topics_ge, alpha_alpha, alpha_beta, lam):
        Inferencer._initialize(self, voc_en, voc_cn, number_of_topics_ge, alpha_alpha, alpha_beta, lam=0.5);
        self._corpus_en = corpus_en
        self._corpus_cn = corpus_cn
        self._trans_en_cn = np.zeros((self._number_of_types_cn, self._number_of_types_en))
        self._trans_cn_en = np.zeros((self._number_of_types_en, self._number_of_types_cn))
        self.parse_data();
        
        # define the total number of document
        self._number_of_documents_en = len(self._word_idss_en)
        self._number_of_documents_cn = len(self._word_idss_cn)
        self._number_of_documents = self._number_of_documents_en + self._number_of_documents_cn
        
        self._n_dk_en = np.zeros((self._number_of_documents_en, self._number_of_topics))
        self._n_dk_cn = np.zeros((self._number_of_documents_cn, self._number_of_topics))
        self._n_kv_en = np.zeros((self._number_of_topics, self._number_of_types_en))
        self._n_kv_cn = np.zeros((self._number_of_topics, self._number_of_types_cn))
        # define the topic assignment for every word in every document, first indexed by doc_id id, then indexed by word word_pos
        self._n_k_en = np.zeros(self._number_of_topics)
        self._n_k_cn = np.zeros(self._number_of_topics)
        self._k_dn_en = {}
        self._k_dn_cn = {}
        self.psi_en = np.zeros((self._number_of_topics, self._number_of_types_en))
        self.psi_cn = np.zeros((self._number_of_topics, self._number_of_types_cn))
        self.phi_en = np.zeros((self._number_of_topics, self._number_of_types_en))
        self.phi_cn = np.zeros((self._number_of_topics, self._number_of_types_cn))
        
        self.random_initialize();

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

        self.psi_en = np.random.gamma(100., 1. / 100., (self._number_of_topics, self._number_of_types_en))
        self.psi_cn = np.random.gamma(100., 1. / 100., (self._number_of_topics, self._number_of_types_cn))
                                    
    def parse_data(self):
        doc_count = 0
        self._word_idss_en = []
        self._word_idss_cn = []
        
        for document_line in self._corpus_en:
            word_ids = [];
            for token in document_line.split():
                if token not in self._type_to_index_en:
                    continue;
                
                type_id = self._type_to_index_en[token];
                if token in self.trans_dict:
                    trans_en_list = self.trans_dict[token]
                    for word_cn in trans_en_list:
                        if word_cn in self._type_to_index_cn:
                            word_id_cn = self._type_to_index_cn[word_cn]
                            self._trans_en_cn[word_id_cn,type_id] += 1
                word_ids.append(type_id);
            if len(word_ids)==0:
                sys.stderr.write("warning: English document collapsed during parsing\n");
                continue;
            self._word_idss_en.append(word_ids);
            doc_count+=1
            if doc_count%1000==0:
                print "successfully parse %d english documents..." % doc_count;
        print "successfully parse %d english documents..." % doc_count
        
        doc_count = 0
        for document_line in self._corpus_cn:
            word_ids = [];
            for token in document_line.split():
                if token not in self._type_to_index_cn:
                    continue;
                type_id = self._type_to_index_cn[token];
                if token in self.trans_dict:
                    trans_cn_list = self.trans_dict[token]
                    for word_en in trans_cn_list:
                        if word_en in self._type_to_index_en:
                            word_id_en = self._type_to_index_en[word_en]
                            self._trans_cn_en[word_id_en, type_id] += 1
                word_ids.append(type_id);
            if len(word_ids)==0:
                sys.stderr.write("warning: Chinese document collapsed during parsing\n");
                continue;
            self._word_idss_cn.append(word_ids);
            doc_count+=1
            if doc_count%1000==0:
                print "successfully parse %d chinese documents..." % doc_count;
        print "successfully parse %d chinese documents..." % (doc_count);

        #normalize translation matrix
        self._trans_cn_en_sum = np.sum(self._trans_cn_en, axis=1)
        t_index = self._trans_cn_en_sum > 0
        self._trans_cn_en[t_index,:] = self._trans_cn_en[t_index,:] / self._trans_cn_en_sum[t_index, np.newaxis]

        self._trans_en_cn_sum = np.sum(self._trans_en_cn, axis=1)
        t_index = self._trans_en_cn_sum > 0
        self._trans_en_cn[t_index,:] = self._trans_en_cn[t_index,:] / self._trans_en_cn_sum[t_index, np.newaxis]
        
    def update_psi(self,):
        term1 = (1-self.lam) * self._n_kv_en / (np.dot(self.psi_cn, self._trans_cn_en.T) * self.lam + (1-self.lam) * self.psi_en)
        term2 = self.lam * np.dot(self._n_kv_cn, self._trans_en_cn) / (np.sum(np.dot(self.psi_en, self._trans_en_cn.T), axis=1) * self.lam + (1-self.lam) * np.sum(self.psi_cn, axis=1))[:, np.newaxis]
        self.psi_en = (term1 + term2) * self.psi_en + self._alpha_beta_en
        self.phi_en = self.lam * np.dot(self.psi_cn, self._trans_cn_en.T) + (1-self.lam) * self.psi_en
        phi_sum = np.sum(self.phi_en, axis=1)
        t_index = phi_sum > 0
        self.phi_en[t_index, :] = self.phi_en[t_index, :] / phi_sum[t_index, np.newaxis]

        term1 = (1-self.lam) * self._n_kv_cn / (np.dot(self.psi_en, self._trans_en_cn.T) * self.lam + (1-self.lam) * self.psi_cn)
        term2 = self.lam * np.dot(self._n_kv_en, self._trans_cn_en) / (np.sum(np.dot(self.psi_cn, self._trans_cn_en.T), axis=1) * self.lam + (1-self.lam) * np.sum(self.psi_en, axis=1))[:, np.newaxis]
        self.psi_cn = (term1 + term2) * self.psi_cn + self._alpha_beta_cn
        self.phi_cn = self.lam * np.dot(self.psi_en, self._trans_en_cn.T) + (1-self.lam) * self.psi_cn
        phi_sum = np.sum(self.phi_cn, axis=1)
        t_index = phi_sum > 0
        self.phi_cn[t_index, :] = self.phi_cn[t_index, :] / phi_sum[t_index, np.newaxis]
        
    """
    this method samples the word at position in document, by covering that word and compute its new topic distribution, in the end, both self._k_dn, self._n_dk and self._n_kv will change
    @param doc_id: a document id
    @param position: the position in doc_id, ranged as range(self._word_idss[doc_id])
    """
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
                log_probability += np.log(self.phi_en[:,word_id])
                
                log_probability -= scipy.misc.logsumexp(log_probability)
                #sample a new topic out of a distribution according to log_probability
                temp_probability = np.exp(log_probability);
                temp_topic_probability = np.random.multinomial(1, temp_probability)[np.newaxis, :]
                new_topic = np.nonzero(temp_topic_probability == 1)[1][0];

                self._n_dk_en[doc_id, new_topic] += 1
                self._n_kv_en[new_topic, word_id] += 1
                self._n_k_en[new_topic] += 1
                self._k_dn_en[doc_id][position] = new_topic

    """
    this method samples the word at position in document, by covering that word and compute its new topic distribution, in the end, both self._k_dn, self._n_dk and self._n_kv will change
    @param doc_id: a document id
    @param position: the position in doc_id, ranged as range(self._word_idss[doc_id])
    """
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
                log_probability += np.log(self.phi_cn[:, word_id])
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
        
        topic_word_probability_en = (self._n_kv_en + self._alpha_beta_en) / (np.sum(self._alpha_beta_en) + np.sum(self._n_kv_en, axis=1))[:, np.newaxis]
        for doc_id in xrange(self._number_of_documents_en):
            doc_topic_probability_en = (self._n_dk_en[doc_id]  + self._alpha_alpha) / (np.sum(self._n_dk_en[doc_id]) + np.sum(self._alpha_alpha))
            word_count_en += len(self._word_idss_en[doc_id])
            for pos in xrange(len(self._word_idss_en[doc_id])):
                word_id = self._word_idss_en[doc_id][pos]
                log_likelihood_en += np.log(np.dot(doc_topic_probability_en, topic_word_probability_en[:, word_id]))

        topic_word_probability_cn = (self._n_kv_cn + self._alpha_beta_cn) / (np.sum(self._alpha_beta_cn) + np.sum(self._n_kv_cn, axis=1))[:, np.newaxis]
        for doc_id in xrange(self._number_of_documents_cn):
            doc_topic_probability_cn = (self._n_dk_cn[doc_id]  + self._alpha_alpha) / (np.sum(self._n_dk_cn[doc_id]) + np.sum(self._alpha_alpha))
            word_count_cn += len(self._word_idss_cn[doc_id])
            for pos in xrange(len(self._word_idss_cn[doc_id])):
                word_id = self._word_idss_cn[doc_id][pos]
                log_likelihood_cn += np.log(np.dot(doc_topic_probability_cn, topic_word_probability_cn[:, word_id]))

        log_likelihood = log_likelihood_en + log_likelihood_cn
        word_count = word_count_en + word_count_cn
        perplexity = np.exp(-1.0 * log_likelihood / word_count)        
        return perplexity
                
    """
    sample the corpus to train the parameters
    @param hyper_delay: defines the delay in updating they hyper parameters, i.e., start updating hyper parameter only after hyper_delay number of gibbs sampling iterations. Usually, it specifies a burn-in period.
    """
    def learning(self):
        #sample the total corpus
        #for iter1 in xrange(number_of_iterations):
        self._counter += 1;
        self.update_psi()
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
        print "iteration %i finished in %d seconds with training perplexity %g" % (self._counter, processing_time, self.perplexity())

    def export_topic_word(self):
        self.update_psi()
        result_dict = {}
        kv_en_ge = self.phi_en
        kv_cn_ge = self.phi_cn
        result_dict['kv_en_ge'] = kv_en_ge
        result_dict['kv_cn_ge'] = kv_cn_ge
        result_dict['type_to_index_en'] = self._type_to_index_en
        result_dict['type_to_index_cn'] = self._type_to_index_cn
        result_dict['index_to_type_en'] = self._index_to_type_en
        result_dict['index_to_type_cn'] = self._index_to_type_cn
        return result_dict
        
    def export_document(self, exp_beta_path):
        output = open(exp_beta_path + '_doc', 'w')
        doc_topic_probability_en = (self._n_dk_en  + self._alpha_alpha) / (np.sum(self._n_dk_en, axis=1) + np.sum(self._alpha_alpha))[:,np.newaxis]
        doc_topic_probability_cn = (self._n_dk_cn  + self._alpha_alpha) / (np.sum(self._n_dk_cn, axis=1) + np.sum(self._alpha_alpha))[:, np.newaxis]
        
        for topic_id in xrange(self._number_of_topics):
            en_doc_index = 0
            cn_doc_index = 0
            en_doc_content = ''
            cn_doc_content = ''
            output.write('========={0}==========\n'.format(topic_id))
            topic_doc_en = doc_topic_probability_en[:, topic_id]
            topic_doc_cn = doc_topic_probability_cn[:, topic_id]            
            
            for doc_id in reversed(np.argsort(topic_doc_en)):
                en_doc_index += 1
                en_doc_content += ' '.join([self._index_to_type_en[x] for x in self._word_idss_en[doc_id]]) + '\n'
                if en_doc_index > 30:
                    output.write(en_doc_content)
                    break

            for doc_id in reversed(np.argsort(topic_doc_cn)):
                cn_doc_index += 1
                cn_doc_content += ' '.join([self._index_to_type_cn[x] for x in self._word_idss_cn[doc_id]]) + '\n'
                if cn_doc_index > 30:
                    output.write(cn_doc_content)
                    break
        output.close()
        
    def export_beta(self, exp_beta_path, top_display=-1):
        output_en = open(exp_beta_path+'_en', 'w')
        output_cn = open(exp_beta_path+'_cn', 'w')
        top_display = 20
        for topic_index in xrange(self._number_of_topics):
            output_en.write("==========\t%d\t==========\n" % (topic_index));
            
            beta_probability_en = self._n_kv_en[topic_index, :] + self._alpha_beta_en;
            beta_probability_en /= np.sum(beta_probability_en);
            
            i = 0;
            for type_index in reversed(np.argsort(beta_probability_en)):
                i += 1;
                output_en.write("%s\t%g\n" % (self._index_to_type_en[type_index], beta_probability_en[type_index]));
                if top_display > 0 and i >= top_display:
                    break;
        output_en.close();

        for topic_index in xrange(self._number_of_topics):
            output_cn.write("==========\t%d\t==========\n" % (topic_index));
            
            beta_probability_cn = self._n_kv_cn[topic_index, :] + self._alpha_beta_cn;
            beta_probability_cn /= np.sum(beta_probability_cn);
            
            i = 0;
            for type_index in reversed(np.argsort(beta_probability_cn)):
                i += 1;
                output_cn.write("%s\t%g\n" % (self._index_to_type_cn[type_index], beta_probability_cn[type_index]));
                if top_display > 0 and i >= top_display:
                    break;
        self.export_document(exp_beta_path)
        
if __name__ == "__main__":
    print "not implemented"
