import argparse
import os
import time
import autokeras as ak
import keras_tuner
import medmnist
import numpy as np
import tensorflow as tf
from medmnist import INFO, Evaluator
from medmnist.info import DEFAULT_ROOT


def main(data_flag, num_trials, input_root, output_root, run, model_path):

    # forzar CPU, sin GPU disponible
    tf.config.set_visible_devices([], 'GPU')

    info = INFO[data_flag]
    _ = getattr(medmnist, info['python_class'])(
        split="train", root=input_root, download=True)

    output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
    if not os.path.isdir(output_root):
        os.makedirs(output_root)

    npz_file = np.load(os.path.join(input_root, "{}.npz".format(data_flag)))

    x_train = npz_file['train_images']
    y_train = npz_file['train_labels']
    x_val   = npz_file['val_images']
    y_val   = npz_file['val_labels']
    x_test  = npz_file['test_images']
    y_test  = npz_file['test_labels']

    # AutoKeras requiere canal explícito: (N,28,28,28) -> (N,28,28,28,1)
    if x_train.ndim == 4:
        x_train = x_train[..., np.newaxis]
        x_val   = x_val[..., np.newaxis]
        x_test  = x_test[..., np.newaxis]

    if model_path is not None:
        model = tf.keras.models.load_model(model_path, custom_objects=ak.CUSTOM_OBJECTS)
        test(model, data_flag, x_train, 'train', output_root, run)
        test(model, data_flag, x_val,   'val',   output_root, run)
        test(model, data_flag, x_test,  'test',  output_root, run)

    if num_trials == 0:
        return

    model = train(data_flag, x_train, y_train, x_val, y_val, num_trials, output_root, run)

    test(model, data_flag, x_train, 'train', output_root, run)
    test(model, data_flag, x_val,   'val',   output_root, run)
    test(model, data_flag, x_test,  'test',  output_root, run)


def train(data_flag, x_train, y_train, x_val, y_val, num_trials, output_root, run):

    clf = ak.ImageClassifier(
        project_name=data_flag,
        metrics=['AUC'],
        objective=keras_tuner.Objective("val_auc", direction="max"),
        overwrite=True,
        max_trials=num_trials,
    )

    clf.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=20,
    )

    model = clf.export_model()

    model_file = os.path.join(output_root, '%s_autokeras_%s' % (data_flag, run))
    try:
        model.save(model_file, save_format="tf")
    except Exception:
        model.save(model_file + '.h5')

    return model


def test(model, data_flag, x, split, output_root, run):

    evaluator = medmnist.Evaluator(data_flag, split)
    y_score = model.predict(x)
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
                        default='./autokeras',
                        type=str)
    parser.add_argument('--num_trials',
                        default=20,
                        help='arquitecturas a explorar; 0 solo evalúa el modelo indicado en --model_path',
                        type=int)
    parser.add_argument('--run',
                        default='model1',
                        help='identificador para nombrar archivos de salida: {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)
    parser.add_argument('--model_path',
                        default=None,
                        help='ruta a modelo preentrenado para evaluar directamente',
                        type=str)

    args = parser.parse_args()

    main(
        args.data_flag,
        args.num_trials,
        args.input_root,
        args.output_root,
        args.run,
        args.model_path,
    )