
# coding: utf-8

# In[192]:


import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.feature import hog
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from scipy.ndimage.measurements import label
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score, classification_report, r2_score, make_scorer, roc_curve, auc,f1_score


# In[193]:


Train = glob.glob('traffic/train/*.jpg')


# In[194]:


print(Train[1])


# In[195]:


def color_hist(img, nbins=32):
    ch1 = np.histogram(img[:,:,0], bins=nbins, range=(0, 256))[0]#We need only the histogram, no bins edges
    ch2 = np.histogram(img[:,:,1], bins=nbins, range=(0, 256))[0]
    ch3 = np.histogram(img[:,:,2], bins=nbins, range=(0, 256))[0]
    hist = np.hstack((ch1, ch2, ch3))
    return hist


# In[196]:


def describe(image, mask = None):
    file_features = []
    for file in image:
        image = mpimg.imread(file)
        hist = cv2.calcHist([image], [0, 1, 2], mask, [8,8,8], [0, 256, 0, 256, 0, 256])
        #cv2.normalize(hist, hist)
        file_features.append(hist.flatten())
    return file_features


# In[197]:


train_ft = describe(Train)

A = np.array(train_ft)

A_scaler = StandardScaler().fit(A)
scaled_A = A_scaler.transform(A)
len(scaled_A)


# In[198]:


train_labels = pd.read_csv('traffic/train.labels',
                          header=None, names =['labels'])
target = []
for i, value in np.ndenumerate(train_labels):
    target.append(value)
j = target
labels =  np.array(j).reshape(len(j),1)


# In[199]:


Test = glob.glob('traffic/test/*.jpg')
print(Test[1])
colorspace = 'YUV'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" 
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off


# In[200]:


test_ft = describe(Test)

B2 = np.array(test_ft)

B_scaler = StandardScaler().fit(B2)
scaled_B1 = A_scaler.transform(B2)
len(scaled_B1)


# In[208]:


EtClass = ExtraTreesClassifier(max_depth=27,
                              max_features= 5,
                              min_samples_leaf= 1,
                              min_samples_split= 6,
                              n_estimators= 1000,
                             class_weight='balanced', random_state=1)
#prediction = SvmClass.predict(scaled_B1)
#print("F1 Score: ", f1_score(testlabels,pred, average='weighted'))


# In[209]:


SvmClass = SVC(kernel='poly',
              C=998144.773878184,
              gamma=0.139581940130327,
              class_weight='balanced', probability=True, random_state=8)

#prediction = SvmClass.predict(scaled_B1)
#print("F1 Score: ", f1_score(testlabels,pred, average='weighted'))


# In[210]:


RndClass = RandomForestClassifier(n_estimators=786, max_depth=46, max_features=5, min_samples_leaf=1, min_samples_split=10,
                                 class_weight='balanced', oob_score=True, n_jobs=2, random_state=8)

#prediction = RndClass.predict(scaled_B1)
#print("F1 Score: ", f1_score(testlabels,pred, average='weighted'))


# In[217]:


Mlp_Class = MLPClassifier(random_state=2,hidden_layer_sizes=(100,220,350,10),max_iter=700,activation= 'relu', learning_rate= 'invscaling', solver='adam')


# In[218]:


from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components = 2)
A = svd.fit_transform(scaled_A,trainlabels)
B2_svd = svd.transform(scaled_B1)


# In[219]:


Mlp_Class.fit(scaled_A,trainlabels)
prediction = mlp_clf.predict(scaled_B1)
file= open("Prediction3.dat","w+")
for k in prediction:
    file.write(str(int(k)) + "\n")
file.close()
#print("F1 Score: ", f1_score(testlabels,pred, average='weighted'))

