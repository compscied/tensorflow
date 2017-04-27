# mnistembryo.py
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# Modified for Embryo Data by DPS2018 Team 2
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]
def extract_images(f):
  print('Extracting', f.name)
  #with gzip.GzipFile(fileobj=f) as bytestream:
  with f as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in EMBRYO image file: %s' % (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data
def dense_to_one_hot(labels_dense, num_classes):
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot
def extract_labels(f, one_hot=False, num_classes=4):
  print('Extracting', f.name)
  with f as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in EMBRYO label file: %s' % (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels
class DataSet(object):
  def __init__(self,images,labels,one_hot=False,dtype=dtypes.float32,reshape=True):
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)
    assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
    self._num_examples = images.shape[0]
    if reshape:
      assert images.shape[3] == 1
      images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
    if dtype == dtypes.float32:
      images = images.astype(numpy.float32)
      images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = self._num_examples + 1 # force shuffle in first batch
  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]
def read_data_sets(train_dir,one_hot=False,dtype=dtypes.float32,reshape=True,validation_size=500):
  TRAIN_IMAGES = '64/Train/Data.dat'
  TRAIN_LABELS = '64/Train/Idx.dat'
  TEST_IMAGES = '64/Test/Data.dat'
  TEST_LABELS = '64/Test/Idx.dat'
  with open(TRAIN_IMAGES, 'rb') as f:
    train_images = extract_images(f)
  with open(TRAIN_LABELS, 'rb') as f:
    train_labels = extract_labels(f, one_hot=one_hot)
  with open(TEST_IMAGES, 'rb') as f:
    test_images = extract_images(f)
  with open(TEST_LABELS, 'rb') as f:
    test_labels = extract_labels(f, one_hot=one_hot)
  if not 0 <= validation_size <= len(train_images):
    raise ValueError('Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), validation_size))
  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]
  train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
  validation = DataSet(validation_images,validation_labels,dtype=dtype,reshape=reshape)
  test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)
  return base.Datasets(train=train, validation=validation, test=test)
