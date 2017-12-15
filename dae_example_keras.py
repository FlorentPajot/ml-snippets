n_folds = 5
training_cols = [] # TODO
target_name = 'target' # TODO

mchk = ModelCheckpoint('PATH', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

for f in range(n_folds):
    print('...fold {}'.format(f))
    d = datetime.datetime.now()
    tb = TensorBoard(log_dir='LOG_PATH', \ 
                       histogram_freq=1, \
                       batch_size=256)
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

    
    if 'model' in locals():
        session = K.get_session()
        for layer in model.layers: 
            for v in layer.__dict__:
                v_arg = getattr(layer,v)
                if hasattr(v_arg,'initializer'):
                    initializer_method = getattr(v_arg, 'initializer')
                    initializer_method.run(session=session)
    else:
        # Parameters
        input_dim = 25 # TODO
        h_dim = 15 # TODO

        # ENCODER
        model = Sequential()
        model.add(Dense(units=h_dim, \
                        input_dim=input_dim, \
                        kernel_initializer=RandomNormal(mean=0.0, stddev=np.sqrt(1/len(training_cols)))))
        model.add(Activation('relu'))

        model.add(Dense(units=h_dim, \
                        kernel_initializer=RandomNormal(mean=0.0, stddev=np.sqrt(1/h_dim))))
        model.add(Activation('relu'))

        # DECODER
    #         model.add(Dense(units=8, \
    #                         kernel_initializer=RandomNormal(mean=0.0, stddev=np.sqrt(1/h_dim))))
    #         model.add(Activation('relu'))

        model.add(Dense(units=input_dim, \
                        kernel_initializer=RandomNormal(mean=0.0, stddev=np.sqrt(1/h_dim))))
        model.add(Activation('linear'))
        
        # Compiling
        from keras.optimizers import SGD, Adam
        sgd = SGD(lr=0.01, decay=5e-6, momentum=0.9, nesterov=True)
    #         adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
        model.compile(loss='mse', \
              optimizer=sgd, \
              metrics=['mape'])
        model.summary()
    
        # Select data for current fols
        x = train_df[training_cols]
        y = train_df[target_name]

        # Rank Gaussian scaling
        print('>> scaling')
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(n_quantiles=10, random_state=11)
        x_scaled = pd.DataFrame(qt.fit_transform(x))

        # Inputing noise
        print('>> inputing noise')
        swn = SwapNoise()
        x_scaled_with_noise = pd.DataFrame(swn.fit_transform(x_scaled.values))

    print('>> spliting')
    x_train, y_train = x_scaled.iloc[list(train_df[train_df.fold != f].index)], x_scaled_with_noise.iloc[list(train_df[train_df.fold != f].index)]
    x_test, y_test = x_scaled.iloc[list(train_df[train_df.fold == f].index)], x_scaled_with_noise.iloc[list(train_df[train_df.fold == f].index)]
    print(np.shape(x_train), np.shape(y_train))
    print(np.shape(x_test), np.shape(y_test))

    print('>> training')
    h = model.fit(x_train.values, y_train.values,
              epochs=50, batch_size=128, verbose=0, callbacks=[tb, lr, mchk],
              validation_data=(x_test.values, y_test.values))
    val_loss.append(h.history['val_loss'][-1])
print('>> final average validation loss: {}'.format(np.mean(val_loss)))