python -m lda.launch_train --en_input_directory=../../input/1_MH370_en_train --ch_input_directory=../../input/1_MH370_cn_train  --output_directory=../../train_output/ --corpus_name=1_MH370 --number_of_topics_ge=30 --training_iterations=100 --lamda=0.5 --alpha_alpha=0.5 --alpha_beta=0.01