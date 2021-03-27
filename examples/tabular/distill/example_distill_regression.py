import logging
from os import path, listdir
from shutil import rmtree
from pathlib import Path
import numpy as np
from datetime import datetime
from autogluon.tabular import TabularDataset, TabularPredictor


TIMEOUT = 600
TIME_LIMIT = None  # set = None to fully train distilled models
PROBLEM_TYPE = 'regression'
LEADERBOARD = f'leaderboard_{PROBLEM_TYPE}.csv'
N_FOLDS = 10

cur_dir = Path(path.dirname((path.realpath(__file__))))
data_dir = cur_dir.parent.parent / 'datasets/UCI_Datasets'

formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
file_handler = logging.FileHandler(cur_dir / 'distillation.log')
file_handler.setFormatter(formatter)
logger = logging.getLogger('logger')
logger.addHandler(console_handler)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

for dataset in sorted(listdir(data_dir)):
# for dataset in ['wine-quality-red', 'yacht']:
    logger.info('=' * 80)
    logger.info(f'Dataset: {dataset}')
    logger.info(f'Problem type: {PROBLEM_TYPE}')
    logger.info(f'Timeout: {TIMEOUT}')
    logger.info(f'Time limit: {TIME_LIMIT}')
    logger.info(f'Number of folds: {N_FOLDS}')
    logger.info('Loading data...')
    data = np.loadtxt(data_dir / dataset / 'data/data.txt')
    with open(data_dir / dataset / 'data/index_target.txt', 'r') as file:
        target = int(file.read())
    logger.info(f'Target: {target}')
    np.random.seed(42)
    folds = np.tile(np.arange(N_FOLDS), data.shape[0] // N_FOLDS + 1)[:data.shape[0]]
    np.random.shuffle(folds)
    for i in range(N_FOLDS):
        logger.info(f'------ Fold {i} ------')
        
        train_indices = folds != i
        test_indices = folds == i
        train_data = TabularDataset(data[train_indices, ])
        test_data = TabularDataset(data[test_indices, ])

        logger.info('Fitting distiller...')
        predictor = TabularPredictor(target, problem_type=PROBLEM_TYPE, eval_metric='r2', verbosity=0)
        predictor.fit(train_data, auto_stack=True, time_limit=TIMEOUT)

        # predictor.delete_models(models_to_keep='best')
        logger.info(f'Best model: {predictor.get_model_best()}')

        logger.info('Distilling knowledge from teacher...')
        distilled_model_names = predictor.distill(time_limit=TIME_LIMIT,
                                                  teacher_preds='soft',
                                                  augment_method=None,
                                                  models_name_suffix='KNOW',
                                                  verbosity=0)

        logger.info('Distilling knowledge from teacher with spunge augmentation...')
        predictor.distill(time_limit=TIME_LIMIT,
                        teacher_preds='hard' if PROBLEM_TYPE == 'multiclass' else 'soft',
                        augment_method='spunge',
                        augment_args={'size_factor': 1},
                        verbosity=0,
                        models_name_suffix='SPUNGE')

        logger.info('Distilling knowledge from teacher with munge augmentation...')
        predictor.distill(time_limit=TIME_LIMIT,
                        teacher_preds='hard' if PROBLEM_TYPE == 'multiclass' else 'soft',
                        augment_method='munge',
                        augment_args={'size_factor': 1},
                        verbosity=0,
                        models_name_suffix='MUNGE')

        logger.info('Fitting students on true labels...')
        predictor.distill(time_limit=TIME_LIMIT,
                          teacher_preds='onehot' if PROBLEM_TYPE == 'multiclass' else None,
                          models_name_suffix='BASE',
                          verbosity=0)

        ldr = predictor.leaderboard(test_data, silent=True)
        ldr['task'] = dataset
        ldr['dataset'] = dataset
        ldr['fold'] = i
        ldr['datetime'] = datetime.now()
        ldr = ldr[['task','dataset','fold','model','score_test','score_val','pred_time_test','pred_time_val',
                   'fit_time','pred_time_test_marginal','pred_time_val_marginal','fit_time_marginal','stack_level',
                   'can_infer','fit_order', 'datetime']]
        logger.info(ldr[['model','score_test','score_val']])

        ldr.to_csv(cur_dir / LEADERBOARD, mode='a', index=False, header=False)

        rmtree(cur_dir.parent.parent.parent / 'AutogluonModels')
