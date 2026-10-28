from keras import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.src.layers import *

from Evaluation import Evaluation_Tracking
from Evaluation_All import seg_evaluation


def c2f_module(x, filters, expansion=0.5):
    """C2f Module (Cross-stage partial with fusion)"""
    hidden_filters = int(filters * expansion)
    x1 = Conv2D(hidden_filters, 1, padding='same', activation='swish')(x)
    x2 = Conv2D(hidden_filters, 1, padding='same', activation='swish')(x)
    y = Concatenate()([x, x1, x2])
    return Conv2D(filters, 1, padding='same', activation='swish')(y)


def sppfc(x):
    """Simplified SPPF Module"""
    pool1 = MaxPooling2D(pool_size=5, strides=1, padding='same')(x)
    pool2 = MaxPooling2D(pool_size=9, strides=1, padding='same')(x)
    pool3 = MaxPooling2D(pool_size=13, strides=1, padding='same')(x)
    x = Concatenate()([x, pool1, pool2, pool3])
    return Conv2D(x.shape[-1], 1, padding='same', activation='swish')(x)


def yolo_v9(input_shape, num_classes, sol):
    inputs = Input(input_shape)

    # Backbone
    x = Conv2D(32, 3, padding='same', activation='swish')(inputs)
    x = Conv2D(64, 3, strides=2, padding='same', activation='swish')(x)
    x = c2f_module(x, 64)
    x = Conv2D(128, 3, strides=2, padding='same', activation='swish')(x)
    x = c2f_module(x, 128)
    x = Conv2D(256, 3, strides=2, padding='same', activation='swish')(x)
    x = c2f_module(x, 256)

    # Neck
    x = sppfc(x)
    x = c2f_module(x, 512)

    # Detection Head
    final_features = x
    x = Conv2D(sol, 1, activation='swish', padding='same')(x)
    x = Conv2D(num_classes + 5, 1, activation='sigmoid')(x)
    detection_head = Reshape((16, 16, num_classes + 5))(x)

    model = Model(inputs=inputs, outputs=[final_features, detection_head])
    return model


def Model_YoloV9(images, target, sol=None):
    if sol is None:
        sol = 64
    input_shape = (256, 256, 3)
    num_classes = 80  # Adjust based on your dataset

    model = yolo_v9(input_shape, num_classes, sol)
    feature_extractor = Model(inputs=model.input, outputs=model.get_layer(index=-2).output)

    features = feature_extractor.predict(images)
    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('yolov9_model.keras', verbose=1, save_best_only=True)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(images, target, validation_split=0.1, batch_size=5, epochs=2,
              callbacks=[earlystopper, checkpointer])

    pred_img = model.predict(images)

    Eval = seg_evaluation(pred_img, target)
    Eval_track = Evaluation_Tracking(pred_img, target)
    return Eval, Eval_track
