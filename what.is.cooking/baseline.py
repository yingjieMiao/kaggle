import functools
import pandas as pd
import tensorflow as tf
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2

BATCH_SIZE = 100
TRAIN_STEPS = 2000
TRAIN_TEST_SPLIT = 0.95

def load_data():
  # Load raw data
  JSON_NAME = '/Users/yingjie/Kaggle.Data/what.is.cooking/train.json'
  data = pd.read_json(JSON_NAME, orient='records');
  data = data.sample(frac=1).reset_index(drop=True)  
    
  # Build dictionary of ingradients
  nrow = data.shape[0]
  voc = set()
  for i in range(nrow):
    voc |= set(data.iloc[i, 2])
  ntrain = int(nrow * TRAIN_TEST_SPLIT)  
  
  # data['ingredients'].values is a list of lists.
  feats = to_serialized_features(data['ingredients'].values)
  labels = data.loc[:, 'cuisine']
  label_voc = list(set(data['cuisine'].values))
  label_voc.sort()
  return feats[:ntrain], labels[:ntrain], feats[ntrain:], labels[ntrain:], voc, label_voc

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

def eval_input_fn(features, feature_columns, labels, batch_size):
  feats = tf.parse_example(
    serialized=features,
    features=tf.feature_column.make_parse_example_spec(feature_columns));
  dataset = tf.data.Dataset.from_tensor_slices((feats, labels))
  dataset = dataset.batch(batch_size)
  return dataset

def pred_input_fn(features, feature_columns, batch_size):
  feats = tf.parse_example(
    serialized=features,
    features=tf.feature_column.make_parse_example_spec(feature_columns));
  dataset = tf.data.Dataset.from_tensor_slices(feats)
  dataset = dataset.batch(batch_size)
  return dataset

def make_predictions(classifier, feature_columns, label_voc, test_x):
  predicts = classifier.predict(
      input_fn=lambda:pred_input_fn(test_x, feature_columns, BATCH_SIZE))
  return [label_voc[pred['class_ids'][0]] for pred in predicts]

def generate_submissions(model_fn):
  data = pd.read_json('/Users/yingjie/Kaggle.Data/what.is.cooking/test.json', orient='records')
  feats = to_serialized_features(data['ingredients'].values)
  predictions = model_fn(feats)
  data['cuisine'] = predictions
  data.to_csv('/Users/yingjie/Kaggle.Data/what.is.cooking/my_submissions.csv',
              columns=['id', 'cuisine'], index=False)

def main(argv):
  train_x, train_y, test_x, test_y, voc, label_voc = load_data()
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
      input_fn=lambda:train_input_fn(train_x, [i_column], train_y, BATCH_SIZE),
      steps=TRAIN_STEPS)
  print('Done')

  eval_result = classifier.evaluate(
      input_fn=lambda:eval_input_fn(test_x, [i_column], test_y, BATCH_SIZE))
  
  print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))  
  
  print('sample predictions:\n');
  predicts = make_predictions(classifier, [i_column], label_voc, test_x[:10])
  for pred, expect in zip(predicts, test_y[:10]):
    print(pred, " vs ", expect)
  
  model_fn = functools.partial(make_predictions, classifier, [i_column], label_voc)
  generate_submissions(model_fn)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

