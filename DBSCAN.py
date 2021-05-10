import tensorflow as tf
# import tensorflow_hub as hub
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from cuml.neighbors import NearestNeighbors
# from sklearn.metrics.pairwise import cosine_similarity
from cuml.feature_extraction.text import TfidfVectorizer
# from sklearn.neighbors import NearestNeighbors
import gc
import pandas as pd
from cuml.cluster import DBSCAN
# from sklearn.cluster import DBSCAN
import numpy as np
import cudf, cupy
import pickle
from textblob import TextBlob

AUTOTUNE = tf.data.experimental.AUTOTUNE
SEED = 100
BATCH_SIZE =128
CHUNK_SIZE = 4096
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256


def preprocessing(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_WIDTH, IMAGE_HEIGHT])
    return image


def load_image(path):
    image = tf.io.read_file(path)
    return preprocessing(image)


def augmentation(ds):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.3),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2)),
        tf.keras.layers.experimental.preprocessing.RandomZoom(0.2, 0.2),
    ])

    ds = ds.batch(BATCH_SIZE)

    ds = ds.map(lambda x: data_augmentation(x))

    return ds.prefetch(1)


def prepare_data(df, augment=False):

    path_ds = tf.data.Dataset.from_tensor_slices(df["image_paths"])
    image_ds = path_ds.map(load_image, num_parallel_calls=AUTOTUNE)

    if augment:
        ds = augmentation(image_ds)
    else:
        ds = image_ds.batch(BATCH_SIZE).prefetch(1)

    return ds


tf.random.set_seed(SEED)
load_dir = '/media/linqing/Others/Kaggle/Shopee/dataset/shopee-product-matching'

df_test = pd.read_csv(load_dir + '/test.csv')
df_test['image_paths'] = load_dir + '/test_images/' + df_test['image']

test_ds = prepare_data(df_test, augment=False)

df_train = pd.read_csv(load_dir + '/train.csv')
df_train['image_paths'] = load_dir + '/train_images/' + df_train['image']

train_ds = prepare_data(df_train, augment=False)

## Embedding


def find_image_index(embeddings, min_dist, **kwargs):

    image_index = []
    potential_index_list = []
    n = embeddings.shape[0] if embeddings.shape[0] < 50 else 50

    knn = NearestNeighbors(n_neighbors=n)  # Find the 50 nearest neighbours.
    knn.fit(embeddings)

    method = kwargs.get("method")
    if method == "DBSCAN_in_knn":   # Apply DBSCAN to each set of similar points with the distance = the shortest distance in each set of KNN neighbours.
        dist, idx = knn.kneighbors(embeddings)
        for d, i in zip(dist, idx):
            min_dist_DBSCAN = min(min_dist, np.min(d[d > 1]))
            dbscan = DBSCAN(eps=min_dist_DBSCAN, min_samples=1)
            fit_index = np.where(dbscan.fit(embeddings[i]).labels_ == 0)
            max_label = np.bincount(dbscan.fit(embeddings[i]).labels_).argmax()

            potential_index = np.where(dbscan.fit(embeddings[i]).labels_ == max_label)
            image_index.append(i[fit_index])
            potential_index_list.append(i[potential_index])

        df_train[method + "img_index"] = image_index

        return image_index, potential_index

    elif method == "DBSCAN_in_distance":
        dist, idx = knn.kneighbors(embeddings)
        dbscan = DBSCAN(eps=1.2, min_samples=1)
        for d, i in zip(dist, idx):
            fit_index = np.where(dbscan.fit(d).labels_ == 0)
            image_index.append(i[fit_index])

        df_train[method + "img_index"] = image_index
        return image_index

    elif method == "KNN_distance":
        dist, idx = knn.kneighbors(embeddings)
        image_counts = (dist < min_dist).sum(axis=1)
        image_index = [idx[i, :image_counts[i]].tolist() for i in range(len(embeddings))]

        df_train[method + "img_index"] = image_index
        return image_index




def eval_f1_score(image_index, df):

    f1_score_list = []
    tp_list = []
    fp_list = []
    fn_list = []

    for y_pred, y_true in zip(image_index, df["matches"]):
        tp = len(np.intersect1d(y_pred, y_true))
        fp = len(y_pred) - tp
        fn = len(y_true) - tp
        f1_score = tp / (tp + (fp + fn)/2)
        f1_score_list.append(f1_score)
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)

    return np.mean(f1_score_list), np.mean(tp_list), np.mean(fp_list), np.mean(fn_list)


with open("train.pkl", "rb") as file:
    df_train = pickle.load(file)


def text_embedding(df_train, threshold, **kwargs):
    method = kwargs.get("method")

    if method == "title_only":
        sentences = cudf.Series(df_train.title)

    elif method == "title_2_noun":
        title_list = []
        for title in df_train.title:
            noun = np.str(TextBlob(title).noun_phrases)
            title_list.append(noun)
        sentences = cudf.Series(title_list)

    vectorizer = TfidfVectorizer(binary=True, max_features=25000)
    embeddings_text = vectorizer.fit_transform(sentences).toarray()

    text_index = []
    text_index_potential = []
    num_chunk = round(embeddings_text.shape[0] / CHUNK_SIZE)
    for i in range(num_chunk + 1):
        start_idx = i * CHUNK_SIZE
        end_idx = min((i + 1) * CHUNK_SIZE, embeddings_text.shape[0])
        print('Chunk', start_idx, 'to', end_idx)

        cts = cupy.matmul(embeddings_text, embeddings_text[start_idx:end_idx].T).T
        for k in range(end_idx - start_idx):
            idx = cupy.where(cts[k,] > 0.72)[0]
            idx_potential = cupy.where(cts[k,] > 0.53)[0]

            text_index.append(cupy.asnumpy(idx))
            text_index_potential.append(cupy.asnumpy(idx_potential))

    # df_train["text_index"] = text_index
    # df_train.to_pickle("train.pkl")

    # text_index = df_train["text_index"]
    return text_index, text_index_potential


def ensemble(test_ds, dist):
    model0 = tf.keras.applications.EfficientNetB0(weights='efficientnetb0_notop.h5', include_top=False, pooling='avg')
    model1 = tf.keras.applications.EfficientNetB1(weights='efficientnetb1_notop.h5', include_top=False, pooling='avg')
    model2 = tf.keras.applications.EfficientNetB2(weights='efficientnetb2_notop.h5', include_top=False, pooling='avg')
    model3 = tf.keras.applications.EfficientNetB3(weights='efficientnetb3_notop.h5', include_top=False, pooling='avg')
    model4 = tf.keras.applications.EfficientNetB4(weights='efficientnetb4_notop.h5', include_top=False, pooling='avg')
    model5 = tf.keras.applications.EfficientNetB5(weights='efficientnetb5_notop.h5', include_top=False, pooling='avg')
    model6 = tf.keras.applications.EfficientNetB6(weights='efficientnetb6_notop.h5', include_top=False, pooling='avg')
    model7 = tf.keras.applications.EfficientNetB7(weights='efficientnetb7_notop.h5', include_top=False, pooling='avg')

    image_indexs = None
    # for model in [model0, model1, model2, model3, model4, model5, model6, model7]:
    for model in [model0]:
        embeddings_image = model.predict(test_ds, verbose=1)
        image_index, potential_index = find_image_index(embeddings_image, dist, method="DBSCAN_in_knn")

        if not image_indexs == None:
            image_indexs = merge_index(image_indexs, image_index)
        else:
            image_indexs = image_index

    return image_indexs, potential_index


def merge_index(index_list0, index_list1):
    item_index = []
    for index0, index1 in zip(index_list0, index_list1):
        item_index.append(list(set(index0) | set(index1)))

    return item_index


def merge_index_potential(index_list0, index_list1):
    item_index = []
    for index0, index1 in zip(index_list0, index_list1):
        item_index.append(list(set(index0) & set(index1)))

    return item_index

#####################################
# model = EfficientNetB0(weights='efficientnetb0_notop.h5', include_top=False, pooling='avg')
# embeddings_image = model.predict(train_ds, verbose=1)
# text_index = text_embedding(df_train)
#
# for dist in np.arange(5.0, 7.0, 0.1):
#     image_index = find_image_index(embeddings_image, dist)
#
#     item_index = []
#     for img, tex in zip(image_index, text_index):
#         item_index.append(list(set(img) | set(tex)))
#     f1_score = eval_f1_score(item_index, df_train)
#
#     print("f1 score = {} for DBSCAN distance = {}".format(f1_score, dist))

######################################
# cts = cupy.matmul(embeddings_text, embeddings_text.T).T
# idx = cupy.where(cts > 0.7)
# text_index.append(cupy.asnumpy(idx))


#####################################

text_index, text_index_potential = text_embedding(df_train, 0.7, method="title_2_noun")
f1_score_text, tp_text, fp_text, fn_text = eval_f1_score(text_index, df_train)
print("f1 score text = {} ".format(f1_score_text))
print("tp text = {} ".format(tp_text))
print("fp text = {} ".format(fp_text))
print("fn text = {} ".format(fn_text))

# b = merge_index_potential(text_index, text_index_potential)

image_index, image_index_potential = ensemble(train_ds, 6.8)
f1_score_img, tp_img, fp_img, fn_img = eval_f1_score(image_index, df_train)
print("f1 score image = {} ".format(f1_score_img))
print("tp image = {}".format(tp_img))
print("fp image = {}".format(fp_img))
print("fn image = {}".format(fn_img))

# text_index = df_train["text_index"]

potential_index = merge_index_potential(image_index_potential, text_index_potential)

item_index = merge_index(image_index, text_index)
f1_score, tp, fp, fn = eval_f1_score(item_index, df_train)
print("f1 score = {} ".format(f1_score))
print("tp = {}".format(tp))
print("fp = {}".format(fp))
print("fn = {}".format(fn))

item_index_ = merge_index(item_index, potential_index)
f1_score_, tp_, fp_, fn_ = eval_f1_score(item_index_, df_train)
print("f1 score final = {} ".format(f1_score_))
print("tp final = {}".format(tp_))
print("fp final = {}".format(fp_))
print("fn final = {}".format(fn_))
a = 1