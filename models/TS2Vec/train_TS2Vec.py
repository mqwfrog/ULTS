import argparse
import os
import time
import datetime
from ts2vec import TS2Vec
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
import matplotlib.pyplot as plt 
import numpy as np 


Class train_TS2Vec():

    def __init__(self, parser):
        self.args = parser.parse_args()
        self.method = 'TS2Vec'
        self.dataset_name  = self.args.dataset_name
        self.run_name = self.args.run_description
        self.archive = self.args.archive
        self.gpu = self.args.device_id
        self.batch_size = self.args.batch_size
        self.lr = self.args.lr
        self.repr_dims = self.args.feature_size
        self.max_train_length = self.args.max_train_length
        self.iters = self.args.iters
        self.epochs = self.args.epochs
        self.save_every = self.args.checkpoint_freq
        self.seed = self.args.seed
        self.max_threads = self.args.max_threads
        self.eval = self.args.eval
        self.irregular = self.args.irregular
    
    def excute(self):
        device = init_dl_program(self.gpu, seed=self.seed, max_threads=self.max_threads)
        
        if self.archive == 'UCR':
            task_type = 'classification'
            train_data, train_labels, test_data, test_labels = datautils.load_UCR(self.dataset)

        elif self.archive == 'UEA':
            print('UEA')
            task_type = 'classification'
            train_data, train_labels, test_data, test_labels = datautils.load_UEA(self.dataset)
        elif self.archive == 'HAR' or 'SHAR' or 'wisdm' or 'epilepsy' or 'sleepEDF':
            print('HAR/SHAR/wisdm/epilepsy')
            task_type = 'classification'
            train_data, train_labels, test_data, test_labels = datautils.load_MTS(self.dataset)

        elif self.archive == 'forecast_csv':
            task_type = 'forecasting'
            data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(self.dataset)
            train_data = data[:, train_slice]
        elif self.archive == 'forecast_csv_univar':
            task_type = 'forecasting'
            data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(self.dataset, univar=True)
            train_data = data[:, train_slice]
        elif self.archive == 'forecast_npy':
            task_type = 'forecasting'
            data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(self.dataset)
            train_data = data[:, train_slice]
        elif self.archive == 'forecast_npy_univar':
            task_type = 'forecasting'
            data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(self.dataset, univar=True)
            train_data = data[:, train_slice]
        else:
            raise ValueError(f"Archive type {self.archive} is not supported.")
            
        if self.irregular > 0:
            if task_type == 'classification':
                train_data = data_dropout(train_data, self.irregular)
                test_data = data_dropout(test_data, self.irregular)
            else:
                raise ValueError(f"Task type {task_type} is not supported when irregular is positive.")
        
        config = dict(
            batch_size=self.batch_size,
            lr=self.lr,
            output_dims=self.repr_dims,
            max_train_length=self.max_train_length
        )
        
        if self.save_every is not None:
            unit = 'epoch' if self.epochs is not None else 'iter'
            config[f'after_{unit}_callback'] = save_checkpoint_callback(self.save_every, unit)

        run_dir = 'training/' + self.dataset + '__' + name_with_datetime(self.run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        t = time.time()
        
        model = TS2Vec(
            input_dims=train_data.shape[-1],
            device=device,
            **config
        )
        loss_log = model.fit(
            train_data,
            n_epochs=self.epochs,
            n_iters=self.iters,
            verbose=True
        )


        train_loss = loss_log
        plt.figure()
        plt.plot(np.arange(self.epochs), train_loss, label="Train")
        plt.title("Loss_%s"%self.dataset)
        plt.legend()
        plt.savefig(os.path.join("./loss/",'loss_%s.png'%self.dataset))
        plt.show()

        model.save(f'{run_dir}/model.pkl')

        t = time.time() - t
        print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

        if self.eval:
            if task_type == 'classification': #mqw dataset=self.dataset,并更改了change-svm
                # out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, dataset=self.dataset, eval_protocol='svm')
                out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, dataset=self.dataset, eval_protocol='linear')
            elif task_type == 'forecasting':
                out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)
            else:
                assert False
            pkl_save(f'{run_dir}/out.pkl', out)
            pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
            print('Evaluation result:', eval_res)

        print("Finished.")

    def save_checkpoint_callback(save_every=1, unit='epoch'):
        assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
        return callback
