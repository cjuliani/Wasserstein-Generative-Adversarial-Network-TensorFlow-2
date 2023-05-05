import os
import config
import argparse
import tensorflow as tf

from src.generator import collect_data, generator
from distutils.util import strtobool
from src.utils.utils import get_gpu_memory
from src.wgan_gp.main import train


bool_fn = lambda x: bool(strtobool(str(x)))  # callable parser type to convert string argument to boolean
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_memory", default=0.5, help='Memory fraction of a GPU used for training.')
parser.add_argument("--gpu_allow_growth", type=bool_fn, default='False', help='Memory growth allowed to GPU.')
parser.add_argument("--gpu_device", default='0', help='Define which GPU device to work on.')
parser.add_argument("--soft_placement", type=bool_fn, default='True',
                    help='Automatically choose an existing device to run tensor operations.')
parser.add_argument("--restore_model", type=str, default=None, help='Restore specified learning model.')
ARGS, unknown = parser.parse_known_args()

# Restrict GPU usage in TensorFlow.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs.
        tf.config.experimental.set_visible_devices(gpus[int(ARGS.gpu_device)], 'GPU')
        for device in gpus:
            try:
                tf.config.experimental.set_memory_growth(device, ARGS.gpu_allow_growth)
            except RuntimeError:
                pass

        # Restrict TensorFlow to only allocate xGB of memory on the first GPU
        gpu_total_memory = get_gpu_memory()[0]
        memory_to_allocate = gpu_total_memory * float(ARGS.gpu_memory)
        tf.config.set_logical_device_configuration(
            gpus[int(ARGS.gpu_device)],
            [tf.config.LogicalDeviceConfiguration(memory_limit=memory_to_allocate)])

        # Let TensorFlow automatically choose an existing and
        # supported device to run the operations (instead of
        # specifying one).
        tf.config.set_soft_device_placement(ARGS.soft_placement)
        tf.print('GPU device used:', gpus[int(ARGS.gpu_device)], 'with memory fraction:', memory_to_allocate)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        raise e

# Run eagerly?
tf.config.run_functions_eagerly(True)


if __name__ == '__main__':
    # Get data from folders and resize.
    data, _ = collect_data(
        path=os.path.join("", "data"),
        output_size=config.INPUT_SIZE)

    # Get generator.
    gen = generator(
        data=data["train"],
        validation_data=data["validation"])

    # Run eagerly.
    tf.config.run_functions_eagerly(True)

    train(
        data_generator=gen,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        checkpoint_model_dir=config.SAVE_PATH,
        valid_step=config.VALID_STEP,
        noise_dim=config.NOISE_DIM,
        augment=config.AUGMENTATION,
        epoch_interval_saving=config.EPOCH_INTERVAL_SAVING,
        img_interval_saving=config.IMG_INTERVAL_SAVING,
        n_critic=config.NUM_CRITIC_STEPS,
        gp_lambda=config.GP_LAMBDA,
        restore_model=ARGS.restore_model)
