#!/usr/bin/python
import cPickle, getopt, sys, time, re
import datetime, os;

import scipy.io;
import nltk;
import numpy as np;
import optparse;

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        en_input_directory=None,
                        ch_input_directory=None,
                        output_directory=None,
        
                        # parameter set 2
                        training_iterations=-1,
                        snapshot_interval=10,
                        number_of_topics_ge=-1,
                        number_of_topics_sp=-1,

                        # parameter set 3
                        alpha_alpha=-1,
                        alpha_beta=-1,
                        
                        # parameter set 4
                        #disable_alpha_theta_update=False,
                        inference_mode=-1,
                        lam = -1.0,)
    # parameter set 1
    parser.add_option("--en_input_directory", type="string", dest="en_input_directory",
                      help="English input directory [None]");
    parser.add_option("--ch_input_directory", type="string", dest="ch_input_directory",
                      help="Chinese input directory [None]")
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]");
    parser.add_option("--corpus_name", type="string", dest="corpus_name",
                      help="the corpus name [None]")
    
    # parameter set 2
    parser.add_option("--number_of_topics_ge", type="int", dest="number_of_topics_ge",
                      help="total number of topics [-1]");
    parser.add_option("--training_iterations", type="int", dest="training_iterations",
                      help="total number of iterations [-1]");
    parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
                      help="snapshot interval [10]");                 
                      
    # parameter set 3
    parser.add_option("--alpha_alpha", type="float", dest="alpha_alpha",
                      help="hyper-parameter for Dirichlet distribution of topics [1.0/number_of_topics]")
    parser.add_option("--alpha_beta", type="float", dest="alpha_beta",
                      help="hyper-parameter for Dirichlet distribution of vocabulary [1.0/number_of_types]")
    parser.add_option("--lamda", type="float", dest="lam",
                      help="the value of lam")
    
    (options, args) = parser.parse_args();
    return options;

def main():
    options = parse_args();

    assert(options.number_of_topics_ge>0);
    number_of_topics_ge = options.number_of_topics_ge;
    number_of_topics = number_of_topics_ge

    assert(options.training_iterations>0);
    training_iterations = options.training_iterations;

    assert(options.snapshot_interval>0);
    if options.snapshot_interval>0:
        snapshot_interval=options.snapshot_interval;
    
    # parameter set 1
    #assert(options.corpus_name!=None);
    assert(options.en_input_directory!=None)
    assert(options.ch_input_directory!=None)
    assert(options.output_directory!=None)
    
    en_input_directory = options.en_input_directory
    ch_input_directory = options.ch_input_directory
    corpus_name = options.corpus_name
    
    output_directory = options.output_directory;
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    output_directory = os.path.join(output_directory, corpus_name);
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    
    # Document
    train_docs_path_en = os.path.join(en_input_directory, 'train.dat')
    input_doc_stream = open(train_docs_path_en, 'r')
    train_docs_en = [];
    for line in input_doc_stream:
        train_docs_en.append(line.strip().lower());
    print "successfully load english training docs from %s..." % (os.path.abspath(train_docs_path_en));

    train_docs_path_cn = os.path.join(ch_input_directory, 'train.dat')
    input_doc_stream = open(train_docs_path_cn, 'r')
    train_docs_cn = [];
    for line in input_doc_stream:
        train_docs_cn.append(line.strip().lower());
    print "successfully load chinese training docs from %s..." % (os.path.abspath(train_docs_path_cn));
        
    # Vocabulary
    vocabulary_path_en = os.path.join(en_input_directory, 'voc.dat')
    input_voc_stream = open(vocabulary_path_en, 'r');
    vocab_en = []
    for line in input_voc_stream:
        vocab_en.append(line.strip().lower().split()[0]);
    vocab_en = list(set(vocab_en));
    print "successfully load english words from %s..." % (os.path.abspath(vocabulary_path_en));

    vocabulary_path_cn = os.path.join(ch_input_directory, 'voc.dat')
    input_voc_stream = open(vocabulary_path_cn, 'r');
    vocab_cn = []
    for line in input_voc_stream:
        vocab_cn.append(line.strip().lower().split()[0]);
    vocab_cn = list(set(vocab_cn));
    print "successfully load chinese words from %s..." % (os.path.abspath(vocabulary_path_cn));

                      
    # parameter set 3
    alpha_alpha = 1.0/number_of_topics;
    if options.alpha_alpha>0:
        alpha_alpha=options.alpha_alpha;
    alpha_beta = options.alpha_beta;
    if options.alpha_beta<=0:
        alpha_beta = 0.01;

    lam = options.lam
    if options.lam <= 0:
        lam = 0.5

    # create output directory
    now = datetime.datetime.now();
    suffix = now.strftime("%y%m%d-%H%M%S") + "";
    suffix += "-%s" % ("crosslda");
    suffix += "-I%d" % (training_iterations);
    suffix += "-S%d" % (snapshot_interval);
    suffix += "-Kge%d" % (number_of_topics_ge);
    suffix += "-aa%f" % (alpha_alpha);
    suffix += "-ab%f" % (alpha_beta);
    suffix += "-lamda%f" % (lam)
    suffix += "/";
    
    output_directory = os.path.join(output_directory, suffix);
    os.mkdir(os.path.abspath(output_directory));

    options_output_file = open(output_directory + "option.txt", 'w');
    # parameter set 1
    options_output_file.write("en_input_directory=" + en_input_directory + "\n");
    options_output_file.write("ch_input_directory=" + ch_input_directory + "\n");                
    options_output_file.write("corpus_name=" + corpus_name + "\n");
    # parameter set 2
    options_output_file.write("training_iterations=%d\n" % (training_iterations));
    options_output_file.write("snapshot_interval=" + str(snapshot_interval) + "\n");
    options_output_file.write("number_of_topics=" + str(number_of_topics) + "\n");
    # parameter set 3
    options_output_file.write("alpha_alpha=" + str(alpha_alpha) + "\n");
    options_output_file.write("alpha_beta=" + str(alpha_beta) + "\n");
    # parameter set 4
    options_output_file.write("lam=%f\n" % (lam))
    options_output_file.close()

    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "output_directory=" + output_directory
    print "en_input_directory=" + en_input_directory
    print "ch_input_directory=" + ch_input_directory
    print "corpus_name=" + corpus_name
    #print "dictionary file=" + str(dict_file)
    # parameter set 2
    print "training_iterations=%d" %(training_iterations);
    print "snapshot_interval=" + str(snapshot_interval);
    print "number_of_topics_ge=" + str(number_of_topics_ge)
    # parameter set 3
    print "alpha_alpha=" + str(alpha_alpha)
    print "alpha_beta=" + str(alpha_beta)
    print "========== ========== ========== ========== =========="
    
    import crosslda
    lda_inferencer = crosslda.CrossLDA();    
    lda_inferencer._initialize(train_docs_en, vocab_en, train_docs_cn, vocab_cn, number_of_topics_ge, alpha_alpha, alpha_beta, lam);
    
    for iteration in xrange(training_iterations):
        lda_inferencer.learning();
        if (lda_inferencer._counter % snapshot_interval == 0):
            lda_inferencer.export_beta(output_directory + 'exp_beta-' + str(lda_inferencer._counter));
            model_snapshot_path = os.path.join(output_directory, 'model-' + str(lda_inferencer._counter));
    r_topic_word = lda_inferencer.export_topic_word()
    model_name = lda_inferencer.__class__.__name__ + '_ge' + str(lda_inferencer._number_of_topics_ge) + '_lamda' + str(lam)
    output_model_directory = os.path.join('../../model/', corpus_name)
    if not os.path.exists(output_model_directory):
        os.makedirs(output_model_directory)
    output_model_path = os.path.join(output_model_directory, model_name)
    cPickle.dump(lda_inferencer.export_topic_word(), open(output_model_path, 'wb'));
    
if __name__ == '__main__':
    main()
