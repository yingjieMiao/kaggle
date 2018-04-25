import pandas as pd
import tensorflow as tf
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2

BATCH_SIZE = 100
TRAIN_STEPS = 2000

def load_data():
  # Load raw data
  JSON_NAME = '/Users/yingjie/Kaggle.Data/what.is.cooking/train.json'
  data = pd.read_json(JSON_NAME, orient='records');
    
  # Build dictionary of ingradients
  nrow = data.shape[0]
  voc = set()
  for i in range(nrow):
    voc |= set(data.iloc[i, 2])
  
  # data['ingredients'].values is a list of lists.
  feats = to_serialized_features(data['ingredients'].values)
  labels = data.loc[:, 'cuisine']
  label_voc = list(set(data['cuisine'].values))
  label_voc.sort()
  return feats, labels, voc, label_voc

def to_tf_example(ingradients):
    return example_pb2.Example(features=feature_pb2.Features(
    feature={
        'ingredients':
            feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
                value=[bytes(x, 'utf-8') for x in ingradients]))
    }))
            
def to_serialized_features(values):
    return [(to_tf_example(x)).SerializeToString() for x in values]            

def train_input_fn(features, feature_columns, labels, batch_size):
  feats = tf.parse_example(
    serialized=features,
    features=tf.feature_column.make_parse_example_spec(feature_columns));
  dataset = tf.data.Dataset.from_tensor_slices((feats, labels))
  dataset = dataset.shuffle(1000).repeat().batch(batch_size)
  return dataset

def main(argv):
  features, labels, voc, label_voc = load_data()
  print('Data loaded.')
  
  c_column = (
      tf.feature_column.categorical_column_with_vocabulary_list(
          key='ingredients',
          vocabulary_list=voc,
          num_oov_buckets=1))
  i_column = tf.feature_column.indicator_column(categorical_column=c_column)
  
  classifier = tf.estimator.LinearClassifier(
      feature_columns=[i_column],
      optimizer=tf.train.AdagradOptimizer(learning_rate=0.05),
      n_classes=20,
      label_vocabulary=label_voc)
  
  print('Training...')
  classifier.train(
      input_fn=lambda:train_input_fn(features, [i_column], labels, BATCH_SIZE),
      steps=TRAIN_STEPS)
  print('Done')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

