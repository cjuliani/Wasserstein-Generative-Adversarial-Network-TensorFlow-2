import os
import tensorflow as tf
import subprocess as sp


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def write_loss_summaries(values, names, writer, step):
    """Write loss summaries to tensorboard."""
    with writer.as_default():
        for name, loss in zip(names, values):
            tf.summary.scalar(name, loss, step=step)


def write_metric_summaries(values, names, writer, step):
    """Write metrics to tensorboard."""
    with writer.as_default():
        for name, val in zip(names, values):
            try:
                tf.summary.scalar(str(name.numpy().decode('ascii')), val, step=step)
            except AttributeError:
                # if run eagerly
                tf.summary.scalar(str(name), val, step=step)


def get_folder_or_create(path, name):
    out_path = os.path.join(path, name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    return out_path
