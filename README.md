

# Shopee - Price Match Guarantee 

https://www.kaggle.com/c/shopee-product-matching

2021-03-10

- Start with [Image Embedding](https://rom1504.medium.com/image-embeddings-ed1b194d113e) 

- Ref:

   https://www.kaggle.com/cdeotte/rapids-cuml-tfidfvectorizer-and-knn/comments

  https://www.kaggle.com/cdeotte/part-2-rapids-tfidfvectorizer-cv-0-700#Compute-Baseline-CV-Score

2021-03-11

- Construct f1-score evaluator
- Try clustering the distance with DBSCAN
- Title embedding (NLP)

2021-03-15

- Opitmized memory usages

2021-03-16

- Try GPU and Implement with RAPIDS
- Tensorflow Hub for text embedding
  - Best score: 0.6986 &#8594; LB: 0.675

2021-03-17

- Ensemble text embedding models with different languages (failed)
- Evaluate best threshold for knn results
  - knn_image: 6.8
  - Ken_text: 0.8

- TfidfVectorizer 
  - max_feature=10000, best score: 0.725
  - max_feature=15000, best score: 0.735
  - max_feature=20000, best score: 0.732
  - max_feature=25000, best score: 0.728

Colab Configurations:

- [Install RAPIDS](https://rapids.ai/)

```python
!git clone https://github.com/rapidsai/rapidsai-csp-utils.git
!bash rapidsai-csp-utils/colab/rapids-colab.sh stable

import sys, os

dist_package_index = sys.path.index('/usr/local/lib/python3.7/dist-packages')
sys.path = sys.path[:dist_package_index] + ['/usr/local/lib/python3.7/site-packages'] + sys.path[dist_package_index:]
sys.path
exec(open('rapidsai-csp-utils/colab/update_modules.py').read(), globals())
```

- Kaggle Data Download

```python
!pip install -q kaggle
from google.colab import files
files.upload()

!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c COMPETITION_NAME
!mkdir data
!unzip -q shopee-product-matching.zip -d ./data
!rm shopee-product-matching.zip
```



- [Restrict Tensorflow usage](https://www.kaggle.com/ragnar123/unsupervised-baseline-arcface) 

```python
# RESTRICT TENSORFLOW TO 2GB OF GPU RAM
# SO THAT WE HAVE 14GB RAM FOR RAPIDS
LIMIT = 2.0
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*LIMIT)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
print('We will restrict TensorFlow to max %iGB GPU RAM'%LIMIT)
print('then RAPIDS can use %iGB GPU RAM'%(16-LIMIT))
```

