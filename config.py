# Training parameters
MODEL_NAME = 'lr1e4_b16_nCrit12_spl125'  # model name
EPOCHS =  10000  # number of training epochs
NOISE_DIM = 100  # noise dimension for image generation
NUM_EXAMPLES_TO_GENERATE = 16  # number of fake images to generate
NUM_CRITIC_STEPS = 12  # step at which training shifts between discriminator/generator (original: 5; speed-up training: 8)
EPOCH_INTERVAL_SAVING = 50  # epoch step when saving train parameters
IMG_INTERVAL_SAVING = 25  # epoch step when saving generated image (fake)
LEARNING_RATE = 1e-4
GP_LAMBDA = 10.  # coefficient of discriminator loss
BATCH_SIZE = 16
INPUT_SIZE = (128, 128)
AUGMENTATION = False
VALID_STEP = 5

# ---
# Directories
LEARNING_MODEL = 'wgan_gp'  # NOTE: do not change, unless new models are written
RESULTS_FOLDER = 'results'  # folder of training results
SUMMARY_PATH = RESULTS_FOLDER + '/summary' + f'/{LEARNING_MODEL}'  # folder path of summary results
SAVE_PATH = RESULTS_FOLDER + '/weights' + f'/{LEARNING_MODEL}'  # folder for checkpoints
