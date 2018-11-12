import numpy as np
import tensorflow as tf
import six
from six.moves.urllib.request import urlretrieve
from abc import ABC, abstractmethod
from keras.layers import Dense, Input
import os 
import keras 
import keras.backend as K


class Discriminator():
    def __init__(self, input_tensor1, input_tensor2, input_dim = None, layers = None, optimizer = None, lr = 0.001, activation = 'relu', name = 'discriminator'):# inpuy tensor placeholder for 
      if input_dim is None and layers is not None:
        warn("Using first layer argument as input dimension for discriminator")
        input_dim = layers[0]
      self.layers = layers
      if self.layers is None: # default is 2 layers of input dimension
        self.layers = [input_dim, input_dim]

      try: 
        self.activation = activation
      except:
        self.activation = 'relu'

      self.name = name
      self.classes = 2
      # USES ADAM WITH SUPPLIED LR, not this optimizer
      self.optimizer = optimizer if optimizer is not None else keras.optimizers.Adam(lr = 0.001)
      self.lr = lr
      #self.sess = K.get_session() 

      self.inp = Input(shape = (input_dim,))
      self.input_tensor1 = input_tensor1
      self.input_tensor2 = input_tensor2
      
      for i in range(len(self.layers)):
        lyr = Dense(self.layers[i], activation = self.activation)
        x1 = lyr(self.input_tensor1 if i == 0 else x1)
        x2 = lyr(self.input_tensor2 if i == 0 else x2)

      self.y = Dense(1 if self.classes == 2 else self.classes,
                activation = "sigmoid" if self.classes == 2 else "softmax")
      self.y1 = self.y(x1)
      self.y2 = self.y(x2)
      self.model = keras.models.Model(inputs = [self.input_tensor1, self.input_tensor2], outputs = [self.y1, self.y2])
      #self.loss = losses.binary_crossentropy
      #self.labels = tf.placeholder(shape = (1,))
      #self.model.compile(loss = self.loss, target_tensors = self.labels) #optimizer = self.optimizer,

      #super(Discriminator, self).__init__(**kwargs)
    #def build(self, input_shape):
      # input = data or latent point
    def __call__(self):
      return [self.y1, self.y2]


def binary_crossentropy(inputs):
    if len(inputs) == 2:
        [mu1, mu2] = inputs
        if isinstance(mu2, list):
            return average([K.binary_crossentropy(mu1, pred) for pred in mu2])
        else:
            return K.binary_crossentropy(mu1, mu2)
    else:
        true = inputs[0]
        return average([K.binary_crossentropy(true, inputs[pred]) for pred in range(1, len(inputs))])


class Dataset(ABC): 
    #def __init__(self, per_label = None):
    #    if per_label is not None
        
    @abstractmethod
    def get_data(self):
        pass

    def shrink_supervised(self, per_label, include_labels = None, shuffle = True):
        data = None
        labels = None
        per_label = int(per_label)
        for i in np.unique(self.y_train[:self.x_train.shape[0]]):
            if include_labels is not None and i in include_labels:
                l = np.squeeze(np.array(np.where(self.y_train[:self.x_train.shape[0]] == i)))
                if shuffle:
                    np.random.shuffle(l)
                if data is not None:
                    data = np.vstack((data, self.x_train[l[:per_label], :]))
                    labels = np.vstack((labels, self.y_train[l[:per_label]]))
                else:
                    print("x_train ", self.x_train.shape, " l shape ", l.shape, " per_label: ", per_label, type(per_label))
                    data = self.x_train[l[:per_label], :]
                    labels = self.y_train[l[:per_label]] 
        self.x_train = data
        self.y_train = labels

def mnist_load(path='mnist.npz'):
    """Loads the MNIST dataset.
    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = get_file(path,
                    origin='https://s3.amazonaws.com/img-datasets/mnist.npz',
                    file_hash='8a61469f7ea1b51cbae51d4f78837e45')
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

class MNIST(Dataset):
    def __init__(self, binary = False, val = 0, per_label = None, path = None):
        self.dims = [28,28]
        self.dim1 = 28
        self.dim2 = 28
        self.dim = 784
        self.binary = binary
        self.name = 'mnist' if not binary else 'binary_mnist'

        self.x_train, self.x_test, self.y_train, self.y_test = self.get_data(path)
        self.x_train = self.x_train[:(self.x_train.shape[0]-val), :]
        self.y_train = self.y_train[:(self.y_train.shape[0]-val)]
        self.x_val = self.x_train[(self.x_train.shape[0]-val):, :]

        if per_label is not None:
            pass

    def get_data(self, onehot = False, path = None):
        if path is None and self.binary:
            path = './datasets/mnist/MNIST_binary'
        elif path is None:
            path = 'MNIST_data/mnist.npz'
        elif self.binary:
            path = path + '/mnist/MNIST_binary'


        def lines_to_np_array(lines):
            return np.array([[int(i) for i in line.split()] for line in lines])

        if not self.binary:
            #from keras.datasets import mnist
            #from tensorflow.keras.datasets.mnist import load_data
            (x_train, y_train), (x_test, y_test) = mnist_load()
            x_train = x_train.astype('float32') / 255.
            x_test = x_test.astype('float32') / 255.
            x_train = x_train.reshape((len(x_train), self.dim))
            x_test = x_test.reshape((len(x_test), self.dim))
            # Clever one hot encoding trick
            if onehot:
                y_train = np.eye(10)[y_train]
                y_test = np.eye(10)[y_test]
            return x_train, x_test, y_train, y_test

        else:
            with open(os.path.join(path, 'binarized_mnist_train.amat')) as f:
                lines = f.readlines()
            train_data = lines_to_np_array(lines).astype('float32')
            with open(os.path.join(path, 'binarized_mnist_valid.amat')) as f:
                lines = f.readlines()
            validation_data = lines_to_np_array(lines).astype('float32')
            with open(os.path.join(path, 'binarized_mnist_test.amat')) as f:
                lines = f.readlines()
            test_data = lines_to_np_array(lines).astype('float32')

            #from tensorflow.keras.datasets.mnist import load_data
            (x_train, y_train), (x_test, y_test) = mnist_load()
            #print(train_data.shape, y_train.shape)
            return train_data, test_data, y_train, y_test
            #return train_data, validation_data, test_data


def get_file(fname,
             origin,
             untar=False,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
    """Downloads a file from a URL if it not already in the cache.
    By default the file at the url `origin` is downloaded to the
    cache_dir `~/.keras`, placed in the cache_subdir `datasets`,
    and given the filename `fname`. The final location of a file
    `example.txt` would therefore be `~/.keras/datasets/example.txt`.
    Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
    Passing a hash will verify the file after download. The command line
    programs `shasum` and `sha256sum` can compute the hash.
    # Arguments
        fname: Name of the file. If an absolute path `/path/to/file.txt` is
            specified the file will be saved at that location.
        origin: Original URL of the file.
        untar: Deprecated in favor of 'extract'.
            boolean, whether the file should be decompressed
        md5_hash: Deprecated in favor of 'file_hash'.
            md5 hash of the file for verification
        file_hash: The expected hash string of the file after download.
            The sha256 and md5 hash algorithms are both supported.
        cache_subdir: Subdirectory under the Keras cache dir where the file is
            saved. If an absolute path `/path/to/folder` is
            specified the file will be saved at that location.
        hash_algorithm: Select the hash algorithm to verify the file.
            options are 'md5', 'sha256', and 'auto'.
            The default 'auto' detects the hash algorithm in use.
        extract: True tries extracting the file as an Archive, like tar or zip.
        archive_format: Archive format to try for extracting the file.
            Options are 'auto', 'tar', 'zip', and None.
            'tar' includes tar, tar.gz, and tar.bz files.
            The default 'auto' is ['tar', 'zip'].
            None or an empty list will return no matches found.
        cache_dir: Location to store cached files, when None it
            defaults to the [Keras Directory](/faq/#where-is-the-keras-configuration-filed-stored).
    # Returns
        Path to the downloaded file
    """  # noqa
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.keras')
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = 'md5'
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.keras')
    datadir = os.path.join(datadir_base, cache_subdir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        pass
        # File found; verify integrity if a hash was provided.
        #if file_hash is not None:
            #if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
            #    print('A local file was found, but it seems to be '
            #          'incomplete or outdated because the ' + hash_algorithm +
            #          ' file hash does not match the original value of ' +
            #          file_hash + ' so we will re-download the data.')
            #    download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath)# dl_progress)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise


    if untar:
        if not os.path.exists(untar_fpath):
            _extract_archive(fpath, datadir, archive_format='tar')
        return untar_fpath

    if extract:
        _extract_archive(fpath, datadir, archive_format)

    return fpath