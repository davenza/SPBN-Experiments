import numpy as np
import pandas as pd
from pgmpy.estimators import GaussianValidationLikelihood




spambase = pd.read_csv('data/Spambase/spambase.csv')
spambase = spambase.drop('class', axis=1)
spambase = spambase.astype(np.float64)

vl = GaussianValidationLikelihood(spambase, seed_fold=0)
vl1 = GaussianValidationLikelihood(spambase, seed_fold=1)


print(vl.local_score('word_freq_make', ['word_freq_address', 'word_freq_all']))
print(vl1.local_score('word_freq_make', ['word_freq_address', 'word_freq_all']))



print(vl.validation_local_score('word_freq_make', ['word_freq_address', 'word_freq_all']))
print(vl1.validation_local_score('word_freq_make', ['word_freq_address', 'word_freq_all']))
