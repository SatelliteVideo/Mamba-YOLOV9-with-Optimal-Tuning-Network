from keras import Model, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Multiply, Dense, Reshape, Add
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import tensorflow.keras.backend as K
from Evaluation import Evaluation_Tracking
from Evaluation_All import seg_evaluation

# -----------------------------
# Channel Attention Block
def channel_attention(x, ratio=8):
    channels = x.shape[-1]
    avg_pool = GlobalAveragePooling2D()(x)
    dense_1 = Dense(channels // ratio, activation='relu')(avg_pool)
    dense_2 = Dense(channels, activation='sigmoid')(dense_1)
    scale = Multiply()([x, tf.expand_dims(tf.expand_dims(dense_2, 1), 1)])
    return scale

# -----------------------------
# Mamba-like Block
def mamba_block(x):
    proj = Conv2D(x.shape[-1], kernel_size=1, activation='gelu', padding='same')(x)
    proj = Conv2D(x.shape[-1], kernel_size=3, activation='gelu', padding='same')(proj)
    return Add()([x, proj])

# -----------------------------
# Optimal Tuning Network
def optimal_tuning(x, filters):
    tuned = Conv2D(filters, 1, activation='relu', padding='same')(x)
    tuned = Conv2D(filters, 1, activation='relu', padding='same')(tuned)
    return tuned

# -----------------------------
# Custom Focal Loss
def binary_focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        pt_1 = tf.where(K.equal(y_true, 1), y_pred, K.ones_like(y_pred))
        pt_0 = tf.where(K.equal(y_true, 0), y_pred, K.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return loss

# -----------------------------
# IoU Loss
def iou_loss(y_true, y_pred):
    pred_box = y_pred[..., 0:4]
    true_box = y_true[..., 0:4]

    pred_xy = pred_box[..., 0:2]
    pred_wh = pred_box[..., 2:4]
    pred_xy1 = pred_xy - pred_wh / 2
    pred_xy2 = pred_xy + pred_wh / 2

    true_xy = true_box[..., 0:2]
    true_wh = true_box[..., 2:4]
    true_xy1 = true_xy - true_wh / 2
    true_xy2 = true_xy + true_wh / 2

    intersect_mins = K.maximum(pred_xy1, true_xy1)
    intersect_maxs = K.minimum(pred_xy2, true_xy2)
    intersect_wh = K.maximum(intersect_maxs - intersect_mins, 0.0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_area = pred_wh[..., 0] * pred_wh[..., 1]
    true_area = true_wh[..., 0] * true_wh[..., 1]
    union_area = pred_area + true_area - intersect_area

    iou = intersect_area / (union_area + K.epsilon())
    return 1.0 - iou

# -----------------------------
# Composite Loss Function
def custom_detection_loss(y_true, y_pred):
    focal = binary_focal_loss()(y_true, y_pred)
    iou = iou_loss(y_true, y_pred)
    return focal + 2.0 * iou

# -----------------------------
# Main FM-YOLOv9-OTN Model
def FM_YOLOv9_OTN(input_shape, num_classes, anchors):
    inputs = Input(input_shape)

    x = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, 3, activation='relu', padding='same')(x)
    x = Conv2D(512, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = channel_attention(x)
    x = mamba_block(x)
    x = optimal_tuning(x, 512)
    final_features = x

    num_anchors = len(anchors)
    detection_head = Conv2D(num_anchors * (5 + num_classes), 1, activation='sigmoid', padding='same')(x)
    detection_head = Reshape((16, 16, num_anchors, 5 + num_classes))(detection_head)

    model = Model(inputs=inputs, outputs=[final_features, detection_head])
    return model

# -----------------------------
# Training + Evaluation Function
def Model_FM_YOLOv9_OTN(images, target):
    input_shape = (256, 256, 3)
    num_classes = 80
    anchors = [1, 2, 3, 4, 5, 6]

    model = FM_YOLOv9_OTN(input_shape, num_classes, anchors)

    feature_extractor = Model(inputs=model.input, outputs=model.get_layer(index=-2).output)
    features = feature_extractor.predict(images)

    earlystopper = EarlyStopping(patience=3, verbose=1)
    checkpointer = ModelCheckpoint('fm_yolov9_otn_model.keras', verbose=1, save_best_only=True)

    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=custom_detection_loss,
                  metrics=['accuracy'])

    model.fit(images, target, validation_split=0.1, batch_size=16, epochs=50,
              callbacks=[earlystopper, checkpointer])

    pred_img = model.predict(images)

    Eval_track = Evaluation_Tracking(pred_img, target)
    Eval = seg_evaluation(pred_img, target)

    return Eval, Eval_track
