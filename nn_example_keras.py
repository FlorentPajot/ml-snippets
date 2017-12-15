training_cols = [] # TODO

train_df = train_df.sample(frac=1).reset_index(drop=True)

train_df_scaled = train_df

h_dim = 24

# INPUT LAYER
model = Sequential()
model.add(BatchNormalization(batch_input_shape=(256, len(training_cols))))

# LAYER 1
model.add(Dense(units=h_dim, \
                input_dim=len(training_cols), \
                kernel_initializer=RandomNormal(mean=0.0, stddev=np.sqrt(1/len(training_cols)))))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
#         model.add(AlphaDropout(0.3, noise_shape=None, seed=None))

# LAYER 2
model.add(Dense(units=h_dim, \
                kernel_initializer=RandomNormal(mean=0.0, stddev=np.sqrt(1/h_dim))))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
#         model.add(AlphaDropout(0.3, noise_shape=None, seed=None))

# LAYER 3
#         model.add(Dense(units=h_dim, \
#                         kernel_initializer=RandomNormal(mean=0.0, stddev=np.sqrt(1/h_dim))))
#         model.add(Activation('relu'))
#         model.add(BatchNormalization())
#         model.add(Dropout(0.1))
#         model.add(AlphaDropout(0.3, noise_shape=None, seed=None))

# LAYER 4
model.add(Dense(units=h_dim, \
                kernel_initializer=RandomNormal(mean=0.0, stddev=np.sqrt(1/h_dim))))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
#     model.add(AlphaDropout(0.3, noise_shape=None, seed=None))

# OUTPUT LAYER 
model.add(Dense(1))
model.compile(loss='mape', \
      optimizer='adam', \
      metrics=['mse'])


model.summary()