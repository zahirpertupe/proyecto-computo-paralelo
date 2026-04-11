import argparse
import os

import flaml
import joblib
import medmnist
import numpy as np
from medmnist import INFO, Evaluator
from medmnist.info import DEFAULT_ROOT


def main(data_flag, time, input_root, output_root, run, model_path):

    time_seconds = time * 60 * 60

    info = INFO[data_flag]
    _ = getattr(medmnist, info['python_class'])(
        split="train", root=input_root, download=True)

    output_root = os.path.join(output_root, data_flag)
    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    npz_file = np.load(os.path.join(input_root, "{}.npz".format(data_flag)))

    x_train = npz_file['train_images']
    y_train = npz_file['train_labels']
    x_val   = npz_file['val_images']
    y_val   = npz_file['val_labels']
    x_test  = npz_file['test_images']
    y_test  = npz_file['test_labels']

    size = x_train[0].size
    X_train = x_train.reshape(x_train.shape[0], size)
    X_val   = x_val.reshape(x_val.shape[0], size)
    X_test  = x_test.reshape(x_test.shape[0], size)

    y_train = y_train.ravel()
    y_val   = y_val.ravel()
    y_test  = y_test.ravel()

    if model_path is not None:
        model = joblib.load(model_path)
        test(model, data_flag, X_train, 'train', output_root, run)
        test(model, data_flag, X_val,   'val',   output_root, run)
        test(model, data_flag, X_test,  'test',  output_root, run)

    if time_seconds == 0:
        return

    model = train(data_flag, time_seconds, X_train, y_train, X_val, y_val, output_root, run)

    test(model, data_flag, X_train, 'train', output_root, run)
    test(model, data_flag, X_val,   'val',   output_root, run)
    test(model, data_flag, X_test,  'test',  output_root, run)


def train(data_flag, time_seconds, X_train, y_train, X_val, y_val, output_root, run):

    automl = flaml.AutoML()

    automl.fit(
        X_train, y_train,
        task='classification',
        time_budget=int(time_seconds),
        X_val=X_val,
        y_val=y_val,
        n_jobs=4,
        log_file_name=os.path.join(output_root, '%s_flaml_%s.log' % (data_flag, run)),
    )

    model_file = os.path.join(output_root, '%s_flaml_%s.m' % (data_flag, run))
    joblib.dump(automl, model_file)

    return automl


def test(model, data_flag, x, split, output_root, run):

    evaluator = medmnist.Evaluator(data_flag, split)
    y_score = model.predict_proba(x)
    auc, acc = evaluator.evaluate(y_score, output_root, run)
    print('%s  auc: %.5f  acc: %.5f' % (split, auc, acc))

    return auc, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_flag',
                        default='fracturemnist3d',
                        type=str)
    parser.add_argument('--input_root',
                        default=DEFAULT_ROOT,
                        type=str)
    parser.add_argument('--output_root',
                        default='./flaml_output',
                        type=str)
    parser.add_argument('--time',
                        default=4,
                        help='tiempo de búsqueda en horas; si es 0 solo evalúa el modelo indicado en --model_path',
                        type=int)
    parser.add_argument('--run',
                        default='model1',
                        help='identificador para nombrar los archivos de salida: {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)
    parser.add_argument('--model_path',
                        default=None,
                        help='ruta a un modelo preentrenado para evaluar directamente',
                        type=str)

    args = parser.parse_args()

    main(
        args.data_flag,
        args.time,
        args.input_root,
        args.output_root,
        args.run,
        args.model_path,
    )