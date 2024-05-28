from tensorflow import keras
from keras.preprocessing import sequence
import numpy as np
from keras.preprocessing.text import Tokenizer

all_in_one_model = keras.models.load_model('model_all.h5')

test_value = ["http://www.bartekbitner.pl/libraries/fof/-/din7", "https://eheadspace.org.au/headspace-centres/murray-bridge/", "http://67.212.168.179/home/bb./www.portalbb.com.br/atualizando/?cli=Cliente&/4NNQFKTCkM/uoAQ0IU3oB.php"]
test_array = np.array(test_value)    

tokener = Tokenizer(lower=True, char_level=True, oov_token='-n-')
tokener.fit_on_texts(test_array)
char_index = tokener.word_index
x_train = np.asanyarray(tokener.texts_to_sequences(test_array))

x_train = sequence.pad_sequences(x_train, maxlen=200) 
# a = all_in_one_model.predict(test_value)
a = all_in_one_model.predict_classes(x_train).tolist()
print(a)
