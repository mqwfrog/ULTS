import math
import os
import numpy
import numpy as np
import torch
import sklearn
import sklearn.svm
import sklearn.externals
import sklearn.model_selection
import joblib 
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score #mqw
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.manifold import TSNE
import utils
import losses
import networks



class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.linear(x)


def plot_confusion_matrix(cm, labels_name, title):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  
    plt.title(title)  
    plt.colorbar() 
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)  
    plt.yticks(num_local, labels_name) 
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for first_index in range(len(cm)):  
        for second_index in range(len(cm[first_index])):
            plt.text(first_index, second_index, cm[first_index][second_index])

def plot_embedding(data, label, title, dataset_name):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min) 
    fig = plt.figure(figsize=(15, 10)) 
    ax = plt.subplot(111)  


    if dataset_name == 'SHAR':  
        for i in range(data.shape[0]):  
            plt.text(data[i, 0], data[i, 1], int(label[i]), color=plt.cm.tab20(label[i] / 20),
                     fontdict={'weight': 'bold', 'size': 7})

        plt.xticks() 
        plt.yticks()
        plt.title(title, fontsize=14)
    else:
        for i in range(data.shape[0]): 
            plt.text(data[i, 0], data[i, 1], int(label[i]), color=plt.cm.tab10(label[i] / 10),
                     fontdict={'weight': 'bold', 'size': 7})
        plt.xticks() 
        plt.yticks()
        plt.title(title, fontsize=14)
    return fig


class TimeSeriesEncoderClassifier(sklearn.base.BaseEstimator,
                                  sklearn.base.ClassifierMixin):
    def __init__(self, compared_length, nb_random_samples, negative_penalty,
                 batch_size, nb_steps, lr, penalty, early_stopping,
                 encoder, params, in_channels, out_channels, cuda=False,
                 gpu=0):
        self.architecture = ''
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.penalty = penalty
        self.early_stopping = early_stopping
        self.encoder = encoder
        self.params = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss = losses.triplet_loss.TripletLoss(
            compared_length, nb_random_samples, negative_penalty
        )
        self.loss_varying = losses.triplet_loss.TripletLossVaryingLength(
            compared_length, nb_random_samples, negative_penalty
        )
        self.classifier = sklearn.svm.SVC()
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)

    def save_encoder(self, prefix_file):
        """
        Saves the encoder and the SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be saved (at '$(prefix_file)_$(architecture)_encoder.pth').
        """
        torch.save(
            self.encoder.state_dict(),
            prefix_file + '_' + self.architecture + '_encoder.pth'
        )

    def save(self, prefix_file):
        """
        Saves the encoder and the SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be saved (at '$(prefix_file)_$(architecture)_classifier.pkl' and
               '$(prefix_file)_$(architecture)_encoder.pth').
        """
        self.save_encoder(prefix_file)
        joblib.dump(  #mqw    sklearn.externals.joblib
            self.classifier,
            prefix_file + '_' + self.architecture + '_classifier.pkl'
        )

    def load_encoder(self, prefix_file):
        """
        Loads an encoder.

        @param prefix_file Path and prefix of the file where the model should
               be loaded (at '$(prefix_file)_$(architecture)_encoder.pth').
        """
        if self.cuda:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=lambda storage, loc: storage.cuda(self.gpu)
            ))
        else:
            self.encoder.load_state_dict(torch.load(
                prefix_file + '_' + self.architecture + '_encoder.pth',
                map_location=lambda storage, loc: storage
            ))

    def load(self, prefix_file):
        """
        Loads an encoder and an SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be loaded (at '$(prefix_file)_$(architecture)_classifier.pkl'
               and '$(prefix_file)_$(architecture)_encoder.pth').
        """
        self.load_encoder(prefix_file)
        self.classifier = joblib.load(
            prefix_file + '_' + self.architecture + '_classifier.pkl'
        ) #sklearn.externals.joblib

    def fit_classifier(self, features, y):
        """
        Trains the classifier using precomputed features. Uses an SVM
        classifier with RBF kernel.

        @param features Computed features of the training set.
        @param y Training labels.
        """
        nb_classes = numpy.shape(numpy.unique(y, return_counts=True)[1])[0]
        print(f'nb_classes:{nb_classes}')
        train_size = numpy.shape(features)[0]
        self.classifier = sklearn.svm.SVC(
            C=1 / self.penalty
            if self.penalty is not None and self.penalty > 0
            else numpy.inf,
            gamma='scale'
        )
        if train_size // nb_classes < 5 or train_size < 50 or self.penalty is not None:
            return self.classifier.fit(features, y)
        else:
            grid_search = sklearn.model_selection.GridSearchCV(
                self.classifier, {
                    'C': [
                        0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000,
                        numpy.inf
                    ],
                    'kernel': ['rbf'],
                    'degree': [3],
                    'gamma': ['scale'],
                    'coef0': [0],
                    'shrinking': [True],
                    'probability': [False],
                    'tol': [0.001],
                    'cache_size': [200],
                    'class_weight': [None],
                    'verbose': [False],
                    'max_iter': [10000000],
                    'decision_function_shape': ['ovr'],
                    'random_state': [None]
                },
                cv=5, n_jobs=5 
            )
            if train_size <= 10000:
                grid_search.fit(features, y)
            else:
                split = sklearn.model_selection.train_test_split(
                    features, y,
                    train_size=10000, random_state=0, stratify=y
                )
                grid_search.fit(split[0], split[2])
            self.classifier = grid_search.best_estimator_
            return self.classifier

    def fit_encoder(self, X, y=None, save_memory=False, verbose=False):
        varying = bool(numpy.isnan(numpy.sum(X)))

        train = torch.from_numpy(X)
        if self.cuda:
            train = train.cuda(self.gpu)

        if y is not None:
            nb_classes = numpy.shape(numpy.unique(y, return_counts=True)[1])[0]
            train_size = numpy.shape(X)[0]
            ratio = train_size // nb_classes

        train_torch_dataset = utils.Dataset(X)
        train_generator = torch.utils.data.DataLoader(
            train_torch_dataset, batch_size=self.batch_size, shuffle=True
        )

        max_score = 0
        i = 0  # Number of performed optimization steps
        epochs = 0  # Number of performed epochs
        count = 0  # Count of number of epochs without improvement
        # Will be true if, by enabling epoch_selection, a model was selected
        # using cross-validation
        found_best = False
        print(f'self.nb_steps:{self.nb_steps}')
        # Encoder training
        while i < self.nb_steps:
            if verbose:
                print('Epoch: ', epochs + 1)
            for batch in train_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                self.optimizer.zero_grad()
                if not varying:
                    loss = self.loss(
                        batch, self.encoder, train, save_memory=save_memory
                    )
                else:
                    loss = self.loss_varying(
                        batch, self.encoder, train, save_memory=save_memory
                    )
                loss.backward()
                self.optimizer.step()
                i += 1
                if i >= self.nb_steps:
                    break
            epochs += 1
            # Early stopping strategy
            if self.early_stopping is not None and y is not None and (
                ratio >= 5 and train_size >= 50
            ):
                # Computes the best regularization parameters
                features = self.encode(X)
                self.classifier = self.fit_classifier(features, y)
                # Cross validation score
                score = numpy.mean(sklearn.model_selection.cross_val_score(
                    self.classifier, features, y=y, cv=5, n_jobs=5
                ))
                count += 1
                # If the model is better than the previous one, update
                if score > max_score:
                    count = 0
                    found_best = True
                    max_score = score
                    best_encoder = type(self.encoder)(**self.params)
                    best_encoder.double()
                    if self.cuda:
                        best_encoder.cuda(self.gpu)
                    best_encoder.load_state_dict(self.encoder.state_dict())
            if count == self.early_stopping:
                break

        # If a better model was found, use it
        if found_best:
            self.encoder = best_encoder

        return self.encoder

    def fit(self, X, y, save_memory=False, verbose=False):
        # Fitting encoder
        self.encoder = self.fit_encoder(
            X, y=y, save_memory=save_memory, verbose=verbose
        )

        # SVM classifier training
        features = self.encode(X)
        self.classifier = self.fit_classifier(features, y)

        return self

    def encode(self, X, batch_size=50):
        print(f'encode')
        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
        self.encoder = self.encoder.eval()

        count = 0
        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    features[
                        count * batch_size: (count + 1) * batch_size
                    ] = self.encoder(batch).cpu()
                    count += 1
            else:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    length = batch.size(2) - torch.sum(
                        torch.isnan(batch[0, 0])
                    ).data.cpu().numpy()
                    features[count: count + 1] = self.encoder(
                        batch[:, :, :length]
                    ).cpu()
                    count += 1

        self.encoder = self.encoder.train()
        return features

    def encode_window(self, X, window, batch_size=50, window_batch_size=10000):
        print(f'encode_window')
        features = numpy.empty((
                numpy.shape(X)[0], self.out_channels,
                numpy.shape(X)[2] - window + 1
        ))
        masking = numpy.empty((
            min(window_batch_size, numpy.shape(X)[2] - window + 1),
            numpy.shape(X)[1], window
        ))
        for b in range(numpy.shape(X)[0]):
            for i in range(math.ceil(
                (numpy.shape(X)[2] - window + 1) / window_batch_size)
            ):
                for j in range(
                    i * window_batch_size,
                    min(
                        (i + 1) * window_batch_size,
                        numpy.shape(X)[2] - window + 1
                    )
                ):
                    j0 = j - i * window_batch_size
                    masking[j0, :, :] = X[b, :, j: j + window]
                features[
                    b, :, i * window_batch_size: (i + 1) * window_batch_size
                ] = numpy.swapaxes(
                    self.encode(masking[:j0 + 1], batch_size=batch_size), 0, 1
                )
        return features

    def predict(self, X, batch_size=50):
        features = self.encode(X, batch_size=batch_size)
        return self.classifier.predict(features)


    def mqw_score(self, X, y, dataset_name, batch_size=50): #mqw-add dataset_name
        """
        Outputs accuracy of the SVM classifier on the given testing data.

        @param X Testing set.
        @param y Testing labels.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        features = self.encode(X, batch_size=batch_size)
        predict_tloss=self.predict(X, batch_size=batch_size)
      
        print('Starting to compute confusion matrix...')
        outs_for_eb = torch.tensor(features).float()
        trgs_for_eb = torch.tensor(y).long()
        predict_tloss = torch.tensor(predict_tloss).long()
    

        cm = confusion_matrix(predict_tloss, trgs_for_eb)  
        print('Starting to save classification performance...')
        r = classification_report(trgs_for_eb, predict_tloss , digits=6, output_dict=True) 
        df = pd.DataFrame(r)  # mqw
        df["cohen"] = cohen_kappa_score(trgs_for_eb, predict_tloss) 
        df.to_csv(f'performance_{dataset_name}.csv', mode='a')  

        ### plot confusion matrix
        print('Starting to plot confusion matrix...')
        # change-13 labels_name
        if dataset_name == 'sleepEDF':
            labels_name = ['Wake (W)', 'Non-rapid eye movement (N1)', 'Non-rapid eye movement (N2)',
                           'Non-rapid eye movement (N3)', 'Rapid Eye Movement (REM)']  # sleepEDF
        elif dataset_name == 'HAR':
            labels_name = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']  # HAR
        elif dataset_name == 'epilepsy':
            labels_name = ['epilepsy', 'not epilepsy']  # epilepsy
        elif dataset_name == 'SHAR':
            labels_name = ['StandingUpFS', 'StandingUpFL', 'Walking', 'Running', 'GoingUpS', 'Jumping', 'GoingDownS',
                           'LyingDownFS', 'SittingDown', 'FallingForw', 'FallingRight', 'FallingBack',
                           'HittingObstacle', 'FallingWithPS', 'FallingBackSC', 'Syncope', 'FallingLeft']  # SHAR
        elif dataset_name == 'wisdm':
            labels_name = ['Walking', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Downstairs']  # wisdm
        plot_confusion_matrix(cm, labels_name, f"{dataset_name}--- Confusion Matrix")
        plt.subplots_adjust(bottom=0.15)
        if not os.path.exists(f'./cm/cm_{dataset_name}'):
            os.makedirs(f'./cm/cm_{dataset_name}')
        plt.savefig(f'./cm/cm_{dataset_name}/{dataset_name}_cm.png', format='png', bbox_inches='tight')
        plt.show()

        ### TSNE Embeddings of representations
        print('Starting to compute t-SNE Embeddings...')
        ts = TSNE(n_components=2, init='pca',
                  random_state=0)  
        result = ts.fit_transform(outs_for_eb)  
        fig = plot_embedding(result, trgs_for_eb,
                             f't-SNE Embeddings of Time Series Representations---Dataset: {dataset_name}', dataset_name)
        if not os.path.exists(f'./eb/eb_{dataset_name}'):
            os.makedirs(f'./eb/eb_{dataset_name}')
        plt.savefig(f'./eb/eb_{dataset_name}/{dataset_name}_pca_2d.png', format='png',
                    bbox_inches='tight') 
        plt.show()

        return self.classifier.score(features, y)


class CausalCNNEncoderClassifier(TimeSeriesEncoderClassifier):
    def __init__(self, compared_length=50, nb_random_samples=10,
                 negative_penalty=1, batch_size=1, nb_steps=2000, lr=0.001,
                 penalty=1, early_stopping=None, channels=10, depth=1,
                 reduced_size=10, out_channels=10, kernel_size=4,
                 in_channels=1, cuda=False, gpu=0):
        super(CausalCNNEncoderClassifier, self).__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping,
            self.__create_encoder(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size, cuda, gpu),
            self.__encoder_params(in_channels, channels, depth, reduced_size,
                                  out_channels, kernel_size),
            in_channels, out_channels, cuda, gpu
        )
        self.architecture = 'CausalCNN'
        self.channels = channels
        self.depth = depth
        self.reduced_size = reduced_size
        self.kernel_size = kernel_size

    def __create_encoder(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size, cuda, gpu):
        encoder = networks.causal_cnn.CausalCNNEncoder(
            in_channels, channels, depth, reduced_size, out_channels,
            kernel_size
        )
        encoder.double()
        if cuda:
            encoder.cuda(gpu)
        return encoder

    def __encoder_params(self, in_channels, channels, depth, reduced_size,
                         out_channels, kernel_size):
        return {
            'in_channels': in_channels,
            'channels': channels,
            'depth': depth,
            'reduced_size': reduced_size,
            'out_channels': out_channels,
            'kernel_size': kernel_size
        }

    def encode_sequence(self, X, batch_size=50):
        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        test = utils.Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1
        )
        length = numpy.shape(X)[2]
        features = numpy.full(
            (numpy.shape(X)[0], self.out_channels, length), numpy.nan
        )
        self.encoder = self.encoder.eval()

        causal_cnn = self.encoder.network[0]
        linear = self.encoder.network[3]

        count = 0
        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    # First applies the causal CNN
                    output_causal_cnn = causal_cnn(batch)
                    after_pool = torch.empty(
                        output_causal_cnn.size(), dtype=torch.double
                    )
                    if self.cuda:
                        after_pool = after_pool.cuda(self.gpu)
                    after_pool[:, :, 0] = output_causal_cnn[:, :, 0]
                    # Then for each time step, computes the output of the max
                    # pooling layer
                    for i in range(1, length):
                        after_pool[:, :, i] = torch.max(
                            torch.cat([
                                after_pool[:, :, i - 1: i],
                                output_causal_cnn[:, :, i: i+1]
                            ], dim=2),
                            dim=2
                        )[0]
                    features[
                        count * batch_size: (count + 1) * batch_size, :, :
                    ] = torch.transpose(linear(
                        torch.transpose(after_pool, 1, 2)
                    ), 1, 2)
                    count += 1
            else:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    length = batch.size(2) - torch.sum(
                        torch.isnan(batch[0, 0])
                    ).data.cpu().numpy()
                    output_causal_cnn = causal_cnn(batch)
                    after_pool = torch.empty(
                        output_causal_cnn.size(), dtype=torch.double
                    )
                    if self.cuda:
                        after_pool = after_pool.cuda(self.gpu)
                    after_pool[:, :, 0] = output_causal_cnn[:, :, 0]
                    for i in range(1, length):
                        after_pool[:, :, i] = torch.max(
                            torch.cat([
                                after_pool[:, :, i - 1: i],
                                output_causal_cnn[:, :, i: i+1]
                            ], dim=2),
                            dim=2
                        )[0]
                    features[
                        count: count + 1, :, :
                    ] = torch.transpose(linear(
                        torch.transpose(after_pool, 1, 2)
                    ), 1, 2)
                    count += 1

        self.encoder = self.encoder.train()
        return features

    def get_params(self, deep=True):
        return {
            'compared_length': self.loss.compared_length,
            'nb_random_samples': self.loss.nb_random_samples,
            'negative_penalty': self.loss.negative_penalty,
            'batch_size': self.batch_size,
            'nb_steps': self.nb_steps,
            'lr': self.lr,
            'penalty': self.penalty,
            'early_stopping': self.early_stopping,
            'channels': self.channels,
            'depth': self.depth,
            'reduced_size': self.reduced_size,
            'kernel_size': self.kernel_size,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'cuda': self.cuda,
            'gpu': self.gpu
        }

    def set_params(self, compared_length, nb_random_samples, negative_penalty,
                   batch_size, nb_steps, lr, penalty, early_stopping,
                   channels, depth, reduced_size, out_channels, kernel_size,
                   in_channels, cuda, gpu):
        self.__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping, channels, depth,
            reduced_size, out_channels, kernel_size, in_channels, cuda, gpu
        )
        return self


class LSTMEncoderClassifier(TimeSeriesEncoderClassifier):
    def __init__(self, compared_length=50, nb_random_samples=10,
                 negative_penalty=1, batch_size=1, nb_steps=2000, lr=0.001,
                 penalty=1, early_stopping=None, in_channels=1, cuda=False,
                 gpu=0):
        super(LSTMEncoderClassifier, self).__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping,
            self.__create_encoder(cuda, gpu), {}, in_channels, 160, cuda, gpu
        )
        assert in_channels == 1
        self.architecture = 'LSTM'

    def __create_encoder(self, cuda, gpu):
        encoder = networks.lstm.LSTMEncoder()
        encoder.double()
        if cuda:
            encoder.cuda(gpu)
        return encoder

    def get_params(self, deep=True):
        return {
            'compared_length': self.loss.compared_length,
            'nb_random_samples': self.loss.nb_random_samples,
            'negative_penalty': self.loss.negative_penalty,
            'batch_size': self.batch_size,
            'nb_steps': self.nb_steps,
            'lr': self.lr,
            'penalty': self.penalty,
            'early_stopping': self.early_stopping,
            'in_channels': self.in_channels,
            'cuda': self.cuda,
            'gpu': self.gpu
        }

    def set_params(self, compared_length, nb_random_samples, negative_penalty,
                   batch_size, nb_steps, lr, penalty, early_stopping,
                   in_channels, cuda, gpu):
        self.__init__(
            compared_length, nb_random_samples, negative_penalty, batch_size,
            nb_steps, lr, penalty, early_stopping, in_channels, cuda, gpu
        )
        return self
