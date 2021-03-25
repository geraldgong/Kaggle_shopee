

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
  - max_feature=10000, best score: 0.725 &#8594; LB: 0.678
  - max_feature=15000, best score: 0.735 &#8594; LB: 0.690
  - max_feature=20000, best score: 0.732
  - max_feature=25000, best score: 0.728

2021-03-22

- Submit with EfficientNetB0 + KNN (image) and TfidVectorizer + cosine similarity (title)
  - LB: 0.702
- Check the possiblity of augmentation of the training image (left-right flip)

2021-03-24

- DBSCAN on 50 nearest distance of image embeddings
  - LB: 0.720

2021-03-25

- Text preprocessing
- Try Indonesian and English pre-trained model embedding
- English --> 18939 samples, Indonesia -->8715 samples, Malay --> 2398 samples, German --> 854 samples



## Colab Configurations:

- Avoid disconnection due to idleness

  Chrome: press **F12**, then run the following JavaScript snippet in the **console**

  ```javascript
  function KeepClicking(){
  console.log("Clicking");45
  document.querySelector("colab-connect-button").click()
  }
  setInterval(KeepClicking,60000)
  ```

  

- [Install RAPIDS on Colab](https://rapids.ai/)

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

Install RAPIDS in conda-env

```bash
conda activate conda-env

conda install -c rapidsai-nightly -c nvidia -c numba -c conda-forge cudf python=3.8 cudatoolkit=11.0
conda install -c rapidsai -c nvidia -c conda-forge -c defaults blazingsql=0.18 cuml=0.18 python=3.8 cudatoolkit=11.0
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



## Evaluation

| Image Embedding                       | Text Embedding                                   | CV score | LB score |
| ------------------------------------- | ------------------------------------------------ | -------- | -------- |
| EfficientNetB0, NearestNeighbor: 0.65 | TifidfVectorizer: 25000, Cosine Similarity: 0.7  | 0.7273   | 0.702    |
| EfficientNetB0, NearestNeighbor: 0.65 | TifidfVectorizer: 15000, Cosine Similarity: 0.7  | 0.7342   | 0.694    |
| EfficientNetB0, NearestNeighbor: 0.65 | TifidfVectorizer: 25000, Cosine Similarity: 0.6  | 0.7377   | 0.688    |
| EfficientNetB0, DBSCAN:  eps= 1       | TifidfVectorizer: 25000, Cosine Similarity: 0.7  | 0.7488   | 0.718    |
| EfficientNetB0, DBSCAN:  eps= 1.2     | TifidfVectorizer: 25000, Cosine Similarity: 0.72 | 0.7498   | 0.720    |
| EfficientNetB0, DBSCAN:  eps= 1.2     | TifidfVectorizer: 25000, DBSCAN: eps=0.13        | 0.7555   | 0.646    |

