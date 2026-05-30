# ---------------- Feed Forward Neural Networks (FFNNs) ------------------------------------ #
from src.config import *


def ffnn(
    X_train,
    X_test,
    X_val,
    y_train,
    y_test,
    y_val,
    save_path=None,
    num_class=8,
    epochs=300,
    batch_size=12
):

    # GPU setup
    physical_devices = tf.config.list_physical_devices('GPU')

    if physical_devices:
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
    else:
        print("No GPU detected, using CPU")

    with tf.device('/GPU:0' if physical_devices else '/CPU:0'):

        # Model
        model = Sequential([
            Dense(99, activation='sigmoid'),
            Dropout(0.2),

            Dense(99, activation='sigmoid'),
            Dropout(0.2),

            Dense(num_class, activation='softmax')
        ])

        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=5,
                verbose=1
            )
        ]

        if save_path:
            callbacks.append(
                ModelCheckpoint(
                    filepath=save_path,
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True,
                    verbose=1
                )
            )

        # Train
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

        # Predict
        prob_distribution = model.predict(X_test)

        # Evaluate
        loss, accuracy = model.evaluate(X_test, y_test)

        # Save final model
        if save_path:
            model.save(save_path)

    return prob_distribution, accuracy




def lstm(
    X_train,
    X_test,
    X_val,
    y_train,
    y_test,
    y_val,
    save_path=None,
    timesteps=50,
    data_dim=9,
    num_class=8,
    epochs=300,
    batch_size=12,
    patience=5
):

    # Reshape data
    X_train = X_train.reshape(X_train.shape[0], timesteps, data_dim)
    X_val = X_val.reshape(X_val.shape[0], timesteps, data_dim)
    X_test = X_test.reshape(X_test.shape[0], timesteps, data_dim)

    # GPU setup
    physical_devices = tf.config.list_physical_devices('GPU')

    if physical_devices:
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
    else:
        print("No GPU detected, using CPU")

    with tf.device('/GPU:0' if physical_devices else '/CPU:0'):

        # Model
        model = Sequential([
            LSTM(data_dim, return_sequences=True,
                 input_shape=(timesteps, data_dim)),

            LSTM(data_dim),

            Dense(num_class, activation='softmax')
        ])

        model.compile(
            loss='categorical_crossentropy',
            optimizer='Adam',
            metrics=['accuracy']
        )

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=patience,
                verbose=1
            )
        ]

        if save_path:
            callbacks.append(
                ModelCheckpoint(
                    filepath=save_path,
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True,
                    verbose=1
                )
            )

        # Train
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            shuffle=True
        )

        # Predict
        prob_distribution = model.predict(X_test)

        # Evaluate
        loss, accuracy = model.evaluate(X_test, y_test)

        #print("LSTM Accuracy:", accuracy)

        # Save model
        if save_path:
            model.save(save_path)

    return prob_distribution, accuracy



def SVM(
    X_train,
    X_test,
    y_train,
    y_test,
    save_path=None,
    C=2,
    kernel='rbf'
):

    # Model
    clf = svm.SVC(
        C=C,
        kernel=kernel,
        probability=True
    )

    # Train
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Probability distribution
    prob_distribution = clf.predict_proba(X_test)

    # Accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)

    print("SVM Accuracy:", accuracy)

    # Save model
    if save_path:
        joblib.dump(clf, save_path)

    return prob_distribution, accuracy


def decision_tree(
    X_train,
    y_train,
    X_test,
    y_test,
    save_path=None
):

    # Convert to numpy
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    clf = DecisionTreeClassifier(
        criterion='gini',
        max_leaf_nodes=8,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42
    )

    # Train
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Probability Distribution
    prob_distribution = pd.concat(
        [
            pd.DataFrame(clf.predict_proba(X_test)),
            pd.Series(y_test, name='Actual').reset_index(drop=True)
        ],
        axis=1
    )

    # Accuracy
    accuracy = metrics.accuracy_score(
        y_test,
        y_pred
    )

    print("Decision Tree Accuracy :", accuracy)

    # Save Model
    if save_path:
        joblib.dump(clf, save_path)

    return accuracy, prob_distribution