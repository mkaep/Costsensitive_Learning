import os
from tensorflow.keras.layers import Conv2D, Activation
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def training_model(max_trace_length, n_activity, num_classes, train_x, train_y, title, class_weights_dic, output_file):
    print("Start training")
    print(class_weights_dic)
    print(train_x.shape)
    print(train_y.shape)
    model = Sequential()
    reg = 0.0001
    # =============================================================================
    #     Input layer
    # =============================================================================
    input_shape = (max_trace_length, n_activity, 2)

    # =============================================================================
    #    Layer 1
    # =============================================================================
    model.add(Conv2D(32, (2, 2),
                     input_shape=input_shape,
                     padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(reg)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # =============================================================================
    #    Layer 2
    # =============================================================================
    model.add(Conv2D(64, (4, 4),
                     padding='same',
                     kernel_regularizer=regularizers.l2(reg), ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # =============================================================================
    #    Layer 3
    # =============================================================================
    #model.add(Conv2D(128, (8, 8),
    #                 padding='same',
    #                 kernel_regularizer=regularizers.l2(reg), ))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    # =============================================================================
    #    Flattening Layer
    # =============================================================================
    model.add(Flatten())

    # =============================================================================
    # Output Layer
    # =============================================================================
    model.add(Dense(num_classes,
                    activation='softmax',
                    name='act_output'))

    model.summary()

    # Configure the model for training
    opt = Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)

    model.compile(loss={'act_output' : 'categorical_crossentropy'}, optimizer=opt, metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=6)

    # Output file
    output_file_path = os.path.join(output_file, title + '.h5')

    # Saving
    model_checkpoint = ModelCheckpoint(output_file_path,
                                       monitor='val_loss',
                                       verbose=0,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       mode='min')

    # Train the model
    print("Train y")
    print(train_y)
    model.fit(train_x, {'act_output': train_y},
              validation_split=0.2,
              verbose=1,
              callbacks=[early_stopping, model_checkpoint],
              batch_size=128,
              epochs=500,
              class_weight=class_weights_dic)

    return model


