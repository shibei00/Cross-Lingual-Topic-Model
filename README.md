# Cross Lingual Topic Model with Unaligned Corpus
A topic model which can identify bilingual topics across unaligned corpus using bilingual dictionary. It is also the implementations of Multilingual Cultural-common Topic Analysis (MCTA), as described in our ACL paper:

Bei Shi, Wai Lam, Lidong Bing and Yinqing Xu. [Detecting Common Discussion Topics Across Culture From News Reader Comments](http://aclweb.org/anthology/P16-1064). In Proceeding of the 54th Annual Meeting of the Association for Computational Linguistics, pp.676-685, 2016. [[.bib]](http://aclweb.org/anthology/P/P16/P16-1064.bib).[[Datasets]](https://github.com/shibei00/Cross-Lingual-Topic-Model/tree/master/input).

## Requirements
1. Python 2.7.8
2. numpy (>= 1.11.3)
3. scipy (>= 0.18.1)
4. nltk (>= 3.0.2)

## Data Format
- It is in the folder of [input](https://github.com/shibei00/Cross-Lingual-Topic-Model/tree/master/input).
- There are 10 events in total. Each event includes Engilish train data, English test data, Chinese train data and Chinese test data. Each line represents a document in the event. We remove the punctuations and the stop words. For English comments, we also stem each word to its root form using Porter Stemmer (Porter, 1980). For the Chinese reader comments, we use the Jieba package to segment and remove Chinese stop words.
- The detail of the data set is shown in the following Table. The number of comments includes both train data and test data.

      | Event Title                 | #English comments | #Chinese comments|
      |-----------------------------|-------------------|------------------|
      | 1 MH370 flight accident     |              8608 |             5223 |
      | 2 ISIS in Iraq              |              6341 |             3263 |
      | 3 Ebola occurs              |              2974 |             1622 |
      | 4 Taiwan Crashed Plane      |              6780 |             2648 |
      | 5 iphone6 publish           |              5837 |             4352 |
      | 6 Shooting of Michael Brown |             17547 |             3693 |
      | 7 Charlie Hebdo shooting    |              1845 |              551 |
      | 8 Shanghai stampede         |              3824 |             3175 |
      | 9 Lee Kuan Yew death        |              2418 |             1534 |
      | 10 AIIB foundation          |              7221 |             3198 |

-  We utilize an English-Chinese dictionary from [MDBG](https://www.mdbg.net/chinese/dictionary?page=cc-cedict)

## Usage

This section describes the usage of the implementations in command line or terminal.

### Training

The source code of train our model is in the folder of `src/train`. Please enter into the folder and run the following command:

`python -m lda.launch_train --en_input_directory=ENGLISH_TRAIN_DATA --ch_input_directory=CHINESE_TRANING_DATA  --output_directory=OUTPUT_DIRECTORY --corpus_name=CORPUS_NAME --number_of_topics_ge=NUMBER_OF_TOPICS --train_iterations=ITERATION_NUMS --lamda=0.5 --alpha_alpha=0.5 --alpha_beta=0.01`

The example of train the event `MH370 flight accident` is shown in the file `run_crosslda_mh370.sh`.

The outputs of the train step include English top words, Chinese top words, top documents related with each topic and the trained model file. The snapshots of the top words and related documents in every 10 iterations are saved in the folder of `train_output/corpus_name`.  The indices of topics in each language are corresponding which also indicates that the common topics are identified. The model file is saved in the folder of `model/corpus_name`.

### Testing and Perplexity Evaluation
We will fit our trained model to the test data of the same event. First, enter into the folder `src/test` and run the following command:

`python -m model_test --test_en_input=ENGLISH_TESTING_DATA --test_cn_input=CHINESE_TESTING_DATA --dict_input=DICTIONARY  --model_path=MODEL_PATH --corpus_name=CORPUS_NAME --test_iterations=ITERATION_NUMS --output_directory=OUTPUT_DIRECTORY`

An example of testing the event `MH370 flight accident` is also provided in `run_test_mh370.sh`. The test model will output the top words and the perplexity value of the event in the foler `test_output/corpus_name`. The lower the perplexity value (CCP) is, the better the performance is.

## Citation 

If you use this code and data, please cite our paper:
```sh
@InProceedings{shi-EtAl:2016:P16-11,
  author    = {Shi, Bei  and  Lam, Wai  and  Bing, Lidong  and  Xu, Yinqing},
  title     = {Detecting Common Discussion Topics Across Culture From News Reader Comments},
  booktitle = {Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month     = {August},
  year      = {2016},
  address   = {Berlin, Germany},
  publisher = {Association for Computational Linguistics},
  pages     = {676--685},
  url       = {http://www.aclweb.org/anthology/P16-1064}
}
```
