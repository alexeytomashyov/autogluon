import logging
from os import path
from shutil import rmtree
from pathlib import Path
from openml import tasks
import numpy as np
from datetime import datetime
from autogluon.tabular import TabularDataset, TabularPredictor


TIMEOUT = 600
TIME_LIMIT = 120  # set = None to fully train distilled models
PROBLEM_TYPE = 'binary'
LEADERBOARD = f'leaderboard_{PROBLEM_TYPE}.csv'

cur_dir = Path(path.dirname((path.realpath(__file__))))

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

selected_tasks = dict(
    binary=[31, 7592, 9952, 9977, 10101, 34539, 146606, 146818, 167120, 168335, 168911, 168912],
    multiclass=[7593, 168329, 168330, 168331, 146195, 167119, 12, 146822, 53]
)

for t in selected_tasks[PROBLEM_TYPE]:
    logger.info('=' * 80)
    oml_task = tasks.get_task(t)
    dataset = oml_task.get_dataset()
    logger.info(f'Task: {t} {dataset.name}')
    logger.info(f'Problem type: {PROBLEM_TYPE}')
    logger.info(f'Timeout: {TIMEOUT}')
    logger.info(f'Time limit: {TIME_LIMIT}')
    logger.info('Loading data...')
    data, _, _, _ = dataset.get_data()
    target = dataset.default_target_attribute
    logger.info(f'Target: {target}')
    for i in range(int(oml_task.estimation_parameters['number_folds'])):
        logger.info(f'------ Fold {i} ------')
        np.random.seed(42)
        train_indices, test_indices = oml_task.get_train_test_split_indices(i)
        train_data = TabularDataset(data.iloc[train_indices, ])
        test_data = TabularDataset(data.iloc[test_indices, ])

        logger.info('Fitting distiller...')
        predictor = TabularPredictor(target, problem_type=PROBLEM_TYPE, eval_metric='accuracy', verbosity=0)
        predictor.fit(train_data, auto_stack=True, time_limit=TIMEOUT)

        # predictor.delete_models(models_to_keep='best')
        logger.info(f'Best model: {predictor.get_model_best()}')

        logger.info('Distilling knowledge from teacher without augmentation...')
        distilled_model_names = predictor.distill(time_limit=TIME_LIMIT,
                                                  teacher_preds='soft',
                                                  augment_method=None,
                                                  models_name_suffix='KNOW',
                                                  verbosity=0)

        if PROBLEM_TYPE != 'multiclass':
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
                          hyperparameters={'GBM': {}, 'CAT': {}, 'XGB': {}, 'RF': {}, 'LR': {}},
                          teacher_preds='onehot' if PROBLEM_TYPE == 'multiclass' else None,
                          models_name_suffix='BASE',
                          verbosity=0)

        ldr = predictor.leaderboard(test_data, silent=True)
        ldr['task'] = t
        ldr['dataset'] = dataset.name
        ldr['fold'] = i
        ldr['datetime'] = datetime.now()
        ldr = ldr[['task','dataset','fold','model','score_test','score_val','pred_time_test','pred_time_val',
                   'fit_time','pred_time_test_marginal','pred_time_val_marginal','fit_time_marginal','stack_level',
                   'can_infer','fit_order', 'datetime']]
        logger.info(ldr[['model','score_test','score_val']])

        ldr.to_csv(cur_dir / LEADERBOARD, mode='a', index=False, header=False)

        rmtree(cur_dir.parent.parent.parent / 'AutogluonModels')
