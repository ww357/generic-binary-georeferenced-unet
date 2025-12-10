import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, BatchNormalization, Activation, Dense, Dropout, 
                                    Lambda, RepeatVector, Reshape, Conv2D, Conv2DTranspose,
                                    MaxPooling2D, GlobalMaxPool2D, concatenate, add, multiply,
                                    AveragePooling2D, UpSampling2D, Concatenate)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import cv2
from tqdm import tqdm
import warnings
import glob 
from datetime import datetime
import json 
from pathlib import Path
warnings.filterwarnings('ignore')
# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

kinit = 'he_normal' # He initializer recommended for layers followed by ReLU activations

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

# is this used?
def bce_dice_loss(y_true, y_pred):
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

# is this used?
def confusion(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg) 
    prec = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return prec, recall

def tp(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth) / (K.sum(y_pos) + smooth) 
    return tp 

def tn(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos 
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth)
    return tn

def prec(y_true, y_pred):
    prec, _ = confusion(y_true, y_pred)
    return prec

def recall(y_true, y_pred):
    _, recall_val = confusion(y_true, y_pred)
    return recall_val

def AttnGatingBlock(x, g, inter_shape, name):
    """Attention Gating Block"""
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(g)

    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same', name='xl'+name)(x)
    shape_theta_x = K.int_shape(theta_x)

    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),
                               strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                               padding='same', name='g_up'+name)(phi_g)

    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same', name='psi'+name)(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)

    upsample_psi = Lambda(lambda x: K.repeat_elements(x, shape_x[3], axis=3), name='psi_up'+name)(upsample_psi)
    y = multiply([upsample_psi, x], name='q_attn'+name)

    result = Conv2D(shape_x[3], (1, 1), padding='same', name='q_attn_conv'+name)(y)
    result_bn = BatchNormalization(name='q_attn_bn'+name)(result)
    return result_bn

def UnetGatingSignal(input, is_batchnorm, name):
    """Gating signal for attention"""
    shape = K.int_shape(input)
    x = Conv2D(shape[3] * 1, (1, 1), strides=(1, 1), padding="same", name=name + '_conv')(input)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_bn')(x)
    x = Activation('relu', name=name + '_act')(x)
    return x

def UnetConv2D(input, outdim, is_batchnorm, name):
    """Basic U-Net convolution block"""
    x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name+'_1')(input)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_1_bn')(x)
    x = Activation('relu', name=name + '_1_act')(x)

    x = Conv2D(outdim, (3, 3), strides=(1, 1), kernel_initializer=kinit, padding="same", name=name+'_2')(x)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_act')(x)
    return x

def ASPP(input, out_channel, name):
    """Atrous Spatial Pyramid Pooling module"""
    # Branch 1: 1x1 convolution
    x1 = Conv2D(out_channel, (1, 1), kernel_initializer=kinit, padding="same", dilation_rate=1, name=name+'_conv1')(input)
    x1 = BatchNormalization(name=name+'_bn1')(x1)
    x1 = Activation('relu', name=name+'_act1')(x1)

    # Branch 2: 3x3 dilation rate 6
    x2 = Conv2D(out_channel, (3, 3), kernel_initializer=kinit, padding="same", dilation_rate=6, name=name+'_conv2')(input)
    x2 = BatchNormalization(name=name+'_bn2')(x2)
    x2 = Activation('relu', name=name+'_act2')(x2)

    # Branch 3: 3x3 dilation rate 12
    x3 = Conv2D(out_channel, (3, 3), kernel_initializer=kinit, padding="same", dilation_rate=12, name=name+'_conv3')(input)
    x3 = BatchNormalization(name=name+'_bn3')(x3)
    x3 = Activation('relu', name=name+'_act3')(x3)

    # Branch 4: 3x3 dilation rate 18
    x4 = Conv2D(out_channel, (3, 3), kernel_initializer=kinit, padding="same", dilation_rate=18, name=name+'_conv4')(input)
    x4 = BatchNormalization(name=name+'_bn4')(x4)
    x4 = Activation('relu', name=name+'_act4')(x4)

    # Branch 5: Global Average Pooling
    x5 = AveragePooling2D(pool_size=(1, 1))(input)
    x5 = Conv2D(out_channel, (1, 1), kernel_initializer=kinit, padding="same", name=name+'_conv5')(x5)
    x5 = BatchNormalization(name=name+'_bn5')(x5)
    x5 = Activation('relu', name=name+'_act5')(x5)
    x5 = UpSampling2D(size=(K.int_shape(input)[1]//K.int_shape(x5)[1], 
                          K.int_shape(input)[2]//K.int_shape(x5)[2]))(x5)

    # Concatenate all branches
    x = Concatenate(axis=3)([x1, x2, x3, x4, x5])
    x = Conv2D(out_channel, (1, 1), kernel_initializer=kinit, padding="same", name=name+'_conv_final')(x)
    x = BatchNormalization(name=name+'_bn_final')(x)
    x = Activation('relu', name=name+'_act_final')(x)
    x = Dropout(0.5)(x)

    return x

def build_attn_unet(input_size, loss_function):
    """Build the complete Attention U-Net with ASPP"""
    inputs = Input(shape=input_size)
    
    # Encoder
    conv1 = UnetConv2D(inputs, 32, is_batchnorm=True, name='conv1')
    conv1 = Dropout(0.2, name='drop_conv1')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = UnetConv2D(pool1, 32, is_batchnorm=True, name='conv2')
    conv2 = Dropout(0.2, name='drop_conv2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = UnetConv2D(pool2, 64, is_batchnorm=True, name='conv3')
    conv3 = Dropout(0.2, name='drop_conv3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = UnetConv2D(pool3, 64, is_batchnorm=True, name='conv4')
    conv4 = Dropout(0.2, name='drop_conv4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Center with ASPP
    center = ASPP(pool4, 128, name='center')
    
    # Decoder with Attention Gates
    g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')
    attn1 = AttnGatingBlock(conv4, g1, 128, '_1')
    up1 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', 
                                     activation='relu', kernel_initializer=kinit)(center), attn1], name='up1')
    
    g2 = UnetGatingSignal(up1, is_batchnorm=True, name='g2')
    attn2 = AttnGatingBlock(conv3, g2, 64, '_2')
    up2 = concatenate([Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', 
                                     activation='relu', kernel_initializer=kinit)(up1), attn2], name='up2')

    g3 = UnetGatingSignal(up2, is_batchnorm=True, name='g3')
    attn3 = AttnGatingBlock(conv2, g3, 32, '_3')
    up3 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', 
                                     activation='relu', kernel_initializer=kinit)(up2), attn3], name='up3')

    up4 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', 
                                     activation='relu', kernel_initializer=kinit)(up3), conv1], name='up4')
    
    # Output layer
    out = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer=kinit, name='final')(up4)
    
    model = Model(inputs=[inputs], outputs=[out])
    return model

print("Model architecture functions loaded successfully")

def load_and_preprocess_image(image_path, patch_size):
    """Load and preprocess a single image for model input"""
    try:
        img = load_img(image_path, color_mode='grayscale')
        original_size = img.size
        img_array = img_to_array(img)
        img_resized = tf.image.resize(img_array, (patch_size, patch_size))
        img_normalized = img_resized.numpy() / 255.0
        return img_normalized, img_array, original_size
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None, None, None

def postprocess_prediction(prediction, original_size, patch_size, threshold):
    binary_mask = (prediction > threshold).astype(np.uint8)
    if original_size != (patch_size, patch_size):
        binary_mask_resized = tf.image.resize(
            binary_mask, original_size[::-1], method='nearest'
        ).numpy()
    else:
        binary_mask_resized = binary_mask
    return binary_mask_resized


def process_batch(image_paths, model, output_dirs, patch_size, threshold, progress_bar=None):
    batch_data = []
    valid_paths = []
    original_sizes = []
    for img_path in image_paths:
        img_processed, _, orig_size = load_and_preprocess_image(img_path, patch_size)
        if img_processed is not None:
            batch_data.append(img_processed)
            valid_paths.append(img_path)
            original_sizes.append(orig_size)
    if not batch_data:
        return []
    batch_array = np.array(batch_data)
    predictions = model.predict(batch_array, verbose=0)
    results = []
    for i, (pred, img_path, orig_size) in enumerate(
        zip(predictions, valid_paths, original_sizes)
    ):
        try:
            filename = Path(img_path).stem
            binary_mask = postprocess_prediction(pred, orig_size, patch_size, threshold)
            mask_path = os.path.join(output_dirs['masks'], f"{filename}_mask.png")
            mask_to_save = (binary_mask.squeeze() * 255).astype(np.uint8)
            cv2.imwrite(mask_path, mask_to_save)
            results.append({
                'filename': filename,
                'original_path': img_path,
                'mask_path': mask_path,
                'processed': True
            })
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            results.append({
                'filename': Path(img_path).stem,
                'original_path': img_path,
                'processed': False,
                'error': str(e)
            })
        if progress_bar:
            progress_bar.update(1)
    return results

################################### NEW ADDED FUNCTIONS ###################################################



# ==================== TRAINING SETUP ====================
def setup_training(patch_size, image_channels):
    """Setup and compile the model"""
    input_size = (patch_size, patch_size, image_channels)
    
    # Build model
    model = build_attn_unet(input_size, dice_loss)
    
    # Compile model
    optimizer = Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss=dice_loss, 
                 metrics=[dsc, tp, tn, prec, recall])
    
    return model

def get_data(training_data_path, patch_size, image_channels, train=True):
    """Load and preprocess images and masks"""
    ids = next(os.walk(training_data_path + "images"))[2]
    X = np.zeros((len(ids), patch_size, patch_size, image_channels), dtype=np.float32)
    
    if train:
        y = np.zeros((len(ids), patch_size, patch_size, 1), dtype=np.float32)
    
    print('Getting and resizing images...')
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        # Load images
        img = load_img(training_data_path + 'images/' + id_, color_mode='grayscale')
        x_img = img_to_array(img)
        x_img = tf.image.resize(x_img, (patch_size, patch_size))
        x_img = x_img.numpy()  # Convert tensor to numpy array

        # Normalize and store image (all 3 channels)
        X[n] = x_img / 255.0  # Store all channels

        # Handle different file extensions for masks
        fname, extension = os.path.splitext(id_)
        mask_id_ = fname + '.png' if extension != '.png' else id_

        # Load masks if training
        if train:
            mask = load_img(training_data_path + 'masks/' + mask_id_, color_mode='grayscale')
            mask = img_to_array(mask)
            mask = tf.image.resize(mask, (patch_size, patch_size))
            mask = mask.numpy()
            y[n] = mask / 255.0  # Normalize mask
    
    print('Done!')
    return (X, y) if train else X

def plot_val_metrics(training_history):
    import matplotlib.pyplot as plt
    # Load training history
    import pickle
    with open(training_history, 'rb') as f:
        history = pickle.load(f)

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['dsc'], label='Training Dice Score')
    plt.plot(history['val_dsc'], label='Validation Dice Score')
    plt.title('Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metric_graph.png')
    plt.show()

    print("Plotting completed!")

def evalation_metrics(model_weights_path, patch_size, image_channels):
    # Load the saved model
    print("Loading trained model...")
    model = build_attn_unet((patch_size, patch_size, image_channels), dice_loss)
    model.load_weights(model_weights_path)

    # Recompile with metrics
    optimizer = Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss=dice_loss, 
                metrics=[dsc, tp, tn, prec, recall])

    # Load TEST data
    print("Loading test data...")
    training_data_path = 'training-data/test/'
    X_test, y_test = get_data(training_data_path, patch_size, image_channels, train=True)

    # Evaluate model
    print("Evaluating model...")
    test_loss, test_dsc, test_tp, test_tn, test_prec, test_recall = model.evaluate(
        X_test, y_test, batch_size=1, verbose=1
    )

    print(f"------------------")
    print(f"Test Results:")
    print(f"Dice Score: {test_dsc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"True Positive Rate: {test_tp:.4f}")
    print(f"True Negative Rate: {test_tn:.4f}")

    # Clear memory
    del X_test, y_test
    tf.keras.backend.clear_session()
    print("Evaluation completed! Memory cleared.")


def visualise_prediction_on_patch(patch_size, image_channels, model_weights_path, test_image):
    """
    Load a new image, predict its mask, and visualise prediction attempt
    """
    image_path = f"training-data/test/images/{test_image}.png" 
    real_mask_path = f"training-data/test/masks/{test_image}.png"
    # Load the trained model
    print("Loading trained model...")
    model = build_attn_unet((patch_size, patch_size, image_channels), dice_loss)
    model.load_weights(model_weights_path)
    
    # Load and preprocess the new image
    print(f"Loading image: {image_path}")
    img = load_img(image_path, color_mode='grayscale')  # RGB for 3-channel input
    original_img = img_to_array(img)  # Keep original for display
    mask = load_img(real_mask_path, color_mode='grayscale')
    origional_mask = img_to_array(mask)
    
    # Preprocess for model input
    img_array = img_to_array(img)
    img_array = tf.image.resize(img_array, (patch_size, patch_size))
    img_array = img_array.numpy() / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    print("Making prediction...")
    prediction = model.predict(img_array, verbose=1)[0]
    binary_mask = (prediction > 0.5).astype(np.uint8)     # Convert prediction to binary mask (0 or 1)
    
    # Visualize results
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_img / 255.0)  # Show original image
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Raw prediction (probability map)
    axes[1].imshow(prediction.squeeze(), cmap='viridis')
    axes[1].set_title('Prediction Probability')
    axes[1].axis('off')
    
    # Binary mask
    axes[2].imshow(binary_mask.squeeze(), cmap='gray')
    axes[2].set_title('Binary Mask (Threshold > 0.5)')
    axes[2].axis('off')
    
    axes[3].imshow(origional_mask.squeeze(), cmap='gray')
    axes[3].set_title('Actual Mask')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()
    
    # Also show overlay
    plt.figure(figsize=(5, 4))
    plt.imshow(original_img / 255.0)  # Background
    plt.imshow(binary_mask.squeeze(), alpha=0.3, cmap='Reds')  # Overlay mask
    plt.title('Overlay: Image + Predicted Mask')
    plt.axis('off')
    plt.show()
    
    return prediction, binary_mask


##########################


