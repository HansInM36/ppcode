import pickle
import numpy as np

x = np.linspace(0,10,1001)

y = np.random.rand(123,123)


f = open('/scratch/ppcode/test/' + 'pickle_test_file', 'wb')
pickle.dump(x, f)
pickle.dump(y, f)
f.close()

f = open('/scratch/ppcode/test/' + 'pickle_test_file', 'rb')
data1 = pickle.load(f)
data2 = pickle.load(f)
f.close()
