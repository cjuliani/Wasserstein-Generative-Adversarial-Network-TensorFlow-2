import os
import time
import config
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

from IPython import display
from src.wgan_gp.model import build_generator, build_critic
from src.utils.utils import write_loss_summaries, get_folder_or_create

# Build models.
generator = build_generator(latent_dim=config.NOISE_DIM, out_activation='sigmoid')
discriminator = build_critic(input_shape=config.INPUT_SIZE + (3,))

# Set optimizers.
gen_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE, 0.5, 0.9)
disc_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE, 0.5, 0.9)

# Generate noise for image generation. To be used to check
# produced images over time.
seed = tf.random.normal([config.NUM_EXAMPLES_TO_GENERATE, config.NOISE_DIM])

image = tf.Variable(
    initial_value=tf.ones((config.BATCH_SIZE, 128, 128, 3)),
    shape=tf.TensorShape((config.BATCH_SIZE, 128, 128, 3)),
    trainable=False)

num_epochs = tf.Variable(0, trainable=False)

# Define summary paths.
summary_dir = get_folder_or_create(path=config.SUMMARY_PATH, name=config.MODEL_NAME)
summary_train_path = os.path.join(summary_dir, "train")

# Define checkpoint object.
checkpoint_dir = get_folder_or_create(path=config.SAVE_PATH, name=config.MODEL_NAME)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=gen_optimizer,
    discriminator_optimizer=disc_optimizer,
    generator=generator,
    discriminator=discriminator,
    num_epoch_used=num_epochs)


def gradient_penalty_loss(gradients):
    """Returns the gradient penalty loss, which is a soft
    version of the Lipschitz constraint. It improves stability
    by penalizing gradients with large norm values."""
    gradients_sqr = tf.square(gradients)
    gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=[1, 2, 3])  # sum over 3 last dimensions of image
    gradients_l2_norm = tf.sqrt(gradients_sqr_sum)
    gradient_penalty = tf.square(1. - gradients_l2_norm)
    return tf.reduce_mean(gradient_penalty)


def discriminator_loss(real_output, generated_output, gradients, lambda_=10):
    """Returns the negative Wasserstein distance between the
    generator distribution and the data distribution, with
    gradient penalty loss."""
    real_loss = -tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(generated_output)
    gp_loss = gradient_penalty_loss(gradients)
    return real_loss + fake_loss + gp_loss * lambda_, gp_loss


def generator_loss(generated_output):
    return -tf.reduce_mean(generated_output)


def random_weighted_average(real_images, fake_images):
    """Returns average between real and fakes images
    (weighted by alpha)."""
    alpha = tf.random.uniform((real_images.shape[0], 1, 1, 1))
    return (alpha * real_images) + ((1 - alpha) * fake_images)


def generate_and_save_images(model, epoch, test_input, img_path):
    """ Saves generated image.
    NOTE:`training` is set to False. This is so all
    layers run in inference mode (batch_norm)."""
    matplotlib.use("Agg")
    predictions = model(test_input, training=False)  # (num_examples, 128, 128, 3)

    _ = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(tf.cast(predictions[i] * 255, tf.int32))
        plt.axis('off')

    img_name = 'image_at_epoch_{:04d}.png'.format(epoch)
    plt.savefig(os.path.join(img_path, img_name), dpi=500, bbox_inches="tight")
    plt.close()


@tf.function
def train_discriminator(images, noise, gp_lambda=10, optimize=True):
    """Optimizes discriminator model and return associated loss."""
    with tf.GradientTape() as disc_tape, tf.GradientTape() as _:
        # Get discriminator image from fake and real images.
        fakes = generator(noise)
        disc_fake = discriminator(fakes)
        disc_real = discriminator(images)

        # Get interpolated image between fake and real
        # discriminations (NOTE: used for gradient penalty).
        interpolated_img = random_weighted_average(images, fakes)
        image.assign(interpolated_img)

        with tf.GradientTape() as nested_tape:
            nested_tape.watch(image)
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            disc_validity_interpolated = discriminator(image)
        nested_gradients = nested_tape.gradient(disc_validity_interpolated, image)

        # Get discriminator loss with gradient penalty.
        disc_loss, gp_loss = discriminator_loss(disc_real, disc_fake, tf.stack(nested_gradients), gp_lambda)

    if optimize:
        # Optimize.
        grad_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))

    return disc_loss, gp_loss


@tf.function
def train_generator(noise, optimize=True):
    """Optimizes generator model and returns associated loss."""
    with tf.GradientTape() as gen_tape:
        fakes = generator(noise)
        generated_output = discriminator(fakes)
        gen_loss = generator_loss(generated_output)

    if optimize:
        grad_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))

    return gen_loss


def train(data_generator, batch_size, epochs, checkpoint_model_dir, valid_step,
          noise_dim, gp_lambda, img_interval_saving=50, epoch_interval_saving=5,
          n_critic=5, augment=False, restore_model=None):
    """Trains generator and discriminator models alternatively given
    n_critic.

    Args:
        data_generator: the data generator.
        batch_size (int): the batch size.
        epochs (int): number of training epochs.
        checkpoint_model_dir (str): directory path to save checkpoints.
        valid_step: step for validation and/or displaying the loss
            results (e.g. if the step is 5, loss and metric values
            are averaged over 5 step prior to being displayed).
        noise_dim (int): noise dimension for image generation.
        gp_lambda (float): coefficient controlling the magnitude of the
            gradient penalty added to the discriminator loss.
        img_interval_saving (int): epoch interval at which a generated
            image is saved (for visual inspection).
        epoch_interval_saving (int): epoch interval at which the model
            weights are saved.
        n_critic (int): step at which training shifts between discriminator
            and generator. 5 in original paper, 8 to speed-up training.
        augment (bool): if True, apply augmentation to data.
        restore_model (str): the directory path of model weights to restore.
    """
    if restore_model is not None:
        # Path to model.
        directory_path = os.path.join(checkpoint_model_dir, restore_model)
        checkpoint.restore(tf.train.latest_checkpoint(directory_path)).expect_partial()
        print(f"Model {restore_model} restored.")

        # Add number of epochs from previously train
        # model to current number of epochs defined
        # for training.
        try:
            ep0 = checkpoint.num_epoch_used
            epochs = [ep0.numpy(), ep0.numpy() + epochs]
            num_epochs.assign(ep0)
            print(f"Epoch index starts from {ep0} given restored model.")
        except AttributeError:
            print("Number of epochs in checkpoint does not exist.")
            epochs = [epochs]
    else:
        epochs = [epochs]

    # Delete previous summary event files from given folder.
    # Useful if training experiments require using same
    # summary output directories.
    try:
        for directory in [summary_train_path]:
            existing_summary_files = os.walk(directory).__next__()[-1]
            if existing_summary_files:
                for file in existing_summary_files:
                    os.remove(os.path.join(directory, file))
    except (PermissionError, StopIteration):
        pass

    # Delete previous summary images.
    try:
        existing_images = os.walk(summary_dir).__next__()[-1]
        existing_images = [img for img in existing_images if img.split(".")[-1] == "png"]
        if existing_images:
            for file in existing_images:
                os.remove(os.path.join(summary_dir, file))
    except (PermissionError, StopIteration):
        pass

    # Create summary writers
    sum_train_writer = tf.summary.create_file_writer(summary_train_path)

    print('\n∎ Training')
    gen_loss = 0.
    num_train_data = (len(data_generator.train_indices) // batch_size) + 1
    for epoch in range(*epochs):
        # Mean losses to display while training given
        # the summary iteration interval.
        avg_losses = tf.zeros(shape=(4,))

        for step in range(num_train_data):
            start = time.time()

            # Generate batch of training data.
            # objects: (B, W, H, F)
            objects = data_generator.next_batch(
                batch_size=batch_size,
                augment=augment)

            # Process training step.
            disc_loss, gp_loss = train_discriminator(
                images=tf.cast(objects, tf.float32),
                noise=tf.random.normal([batch_size, noise_dim]),
                gp_lambda=tf.constant(gp_lambda),
                optimize=tf.constant(True))

            if step % n_critic == 0:
                gen_loss = train_generator(
                    noise=tf.random.normal([batch_size, noise_dim]),
                    optimize=tf.constant(True))

            # Make loss and metric vectors.
            loss_vector = tf.concat([[gen_loss + disc_loss], [gen_loss], [disc_loss], [gp_loss]], axis=0)
            loss_names = ['total_loss', 'gen_loss', 'disc_loss', 'gp_loss']

            avg_losses += loss_vector / valid_step

            # Write summaries with global step.
            global_step = step + (epoch * num_train_data)
            write_loss_summaries(
                values=loss_vector,
                names=loss_names,
                writer=sum_train_writer,
                step=tf.cast(global_step, tf.int64))

            # Measure training loop execution time
            end = time.time()
            speed = round(end - start, 2)

            # Display results at given interval.
            if (step % valid_step) == 0:
                if step == 0:
                    avg_losses = loss_vector

                # Display current losses.
                text = f"{step + 1}/{num_train_data} (epoch: {epoch + 1}): "
                text += 'total_loss: {:.3f} - gen_loss: {:.3f} - disc_loss: {:.3f}'.format(*avg_losses)
                text += f" ({speed} sec)"
                tf.print(text)

                # Reset loss vector and metric states.
                avg_losses = tf.zeros(shape=(4,))

        if ((epoch % epoch_interval_saving) == 0) and (epoch != 0):
            # Save the model every x epochs.
            checkpoint.save(file_prefix=checkpoint_prefix)

        if (epoch % img_interval_saving) == 0:
            # Produce images for the GIF as you go.
            display.clear_output(wait=True)
            generate_and_save_images(
                generator, epoch + 1,
                test_input=seed,
                img_path=summary_dir)

        # Increment number of epochs passed.
        num_epochs.assign_add(1)
