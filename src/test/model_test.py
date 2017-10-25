#-*- coding:utf-8 -*-
import os
import datetime
import cPickle as pickle
import optparse
import mc_eval

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(
        test_en_input = None,
        test_cn_input = None,
        dict_input = None,
        model_path = None,
        corpus_name = None,
        test_iterations=-1,
    )

    parser.add_option("--test_en_input", type="string", dest="test_en_input", help="English test documents [None]")
    parser.add_option("--test_cn_input", type="string", dest="test_cn_input", help="Chinese test documents [None]")
    parser.add_option("--dict_input", type="string", dest="dict_input", help="Dictionary [None]")
    parser.add_option("--model_path", type="string", dest="model_path", help="Model [None]")
    parser.add_option("--corpus_name", type="string", dest="corpus_name", help="Corpus Name [None]")    
    parser.add_option("--test_iterations", type="int", dest="test_iterations", help="total number of test iterations [-1]")
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]");

    (options, args) = parser.parse_args();
    return options    

def main():
    options = parse_args()
    assert(options.test_iterations>0)
    test_en_input = options.test_en_input
    test_cn_input = options.test_cn_input
    dict_input = options.dict_input
    model_path = options.model_path
    model_name = os.path.basename(model_path)
    corpus_name = options.corpus_name
    test_iterations = options.test_iterations
    output_directory = options.output_directory

    # Document
    test_docs_path_en = os.path.join(test_en_input, 'test.dat')
    test_docs_en = []
    test_docs_stream = open(test_docs_path_en, 'r')
    for line in test_docs_stream:
        test_docs_en.append(line.strip().lower())
    print "successfully load english test docs from %s..." % (os.path.abspath(test_docs_path_en))
    test_docs_path_cn = os.path.join(test_cn_input, 'test.dat')
    test_docs_cn = []
    test_docs_stream = open(test_docs_path_cn, 'r')
    for line in test_docs_stream:
        test_docs_cn.append(line.strip().lower())
    print "successfully load chinese test docs from %s..." % (os.path.abspath(test_docs_path_cn))

    # create output directory
    output_directory = options.output_directory;
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);
    output_directory = os.path.join(output_directory, corpus_name);
    if not os.path.exists(output_directory):
        os.mkdir(output_directory);

    now = datetime.datetime.now();
    suffix = now.strftime("%y%m%d-%H%M%S") + "";
    suffix += "-%s" % ("crosslda");
    suffix += "-I%d" % (test_iterations);
    suffix += "-%s" % (corpus_name);
    suffix += "/";

    output_directory = os.path.join(output_directory, suffix);
    if not os.path.exists(output_directory):
        os.makedirs(os.path.abspath(output_directory));

    topic_directory = os.path.join(output_directory, 'topics/')
    if not os.path.exists(topic_directory):
        os.makedirs(os.path.abspath(topic_directory));

    print "========== ========== ========== ========== =========="
    # parameter set 1
    print "test_en_input=" + test_en_input
    print "test_cn_input=" + test_cn_input
    print "dict_input=" + dict_input
    print "model_path=" + model_path
    print "corpus_name=" + corpus_name
    print "test_iterations=" + str(test_iterations)
    print "========== ========== ========== ========== =========="

    model = pickle.load(open(model_path, 'rb'))
    mc_test =  mc_eval.MC_EVAL(test_docs_en, test_docs_cn, model, model_name, corpus_name,)
    for iteration in xrange(test_iterations):
        mc_test.learning()
    mc_test.test_model(topic_directory)
    mc_test.output_result(output_directory)

if __name__=="__main__":
    main()
            
