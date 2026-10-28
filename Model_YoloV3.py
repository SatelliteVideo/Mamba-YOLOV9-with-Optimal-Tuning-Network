from keras import Model
from keras.src.layers import *
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from keras.src.optimizers import Adam
from Evaluation import Evaluation_Tracking
from Evaluation_All import seg_evaluation




def yolo_v3(input_shape, num_classes, anchors, sol):
    inputs = Input(input_shape)

    # Backbone
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
    x = Conv2D(sol, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    final_features = x  # Extract the final feature map before detection

    num_anchors = len(anchors)
    detection_head = Conv2D(num_anchors * (5 + num_classes), 1, activation='linear', padding='same')(x)
    detection_head = Reshape((16, 16, num_anchors, 5 + num_classes))(detection_head)

    model = Model(inputs=inputs, outputs=[final_features, detection_head])
    return model


def Model_YoloV3(images, target, sol=None):
    if sol is None:
        sol = 5
    input_shape = (256, 256, 3)
    num_classes = 80
    anchors = [1, 2, 3, 4, 5, 6]  # Example anchor boxes

    model = yolo_v3(input_shape, num_classes, anchors, sol)

    feature_extractor = Model(inputs=model.input, outputs=model.get_layer(index=-2).output)  # Extract final features

    features = feature_extractor.predict(images)
    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('yolov7_model.keras', verbose=1, save_best_only=True)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(images, target, validation_split=0.1, batch_size=5, epochs=2,
              callbacks=[earlystopper, checkpointer])

    # Make predictions
    pred_img = model.predict(images)

    # Evaluation (you would need to implement the Segmentation_Evaluation function)
    Eval = seg_evaluation(pred_img, target)
    Eval_track = Evaluation_Tracking(pred_img, target)
    return Eval, Eval_track

