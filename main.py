# This is a sample Python script.
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
import matplotlib.pyplot as plt

def obtain_scaled_data():
    housing = fetch_california_housing()
    x_tr_, x_ts, y_tr_, y_ts = train_test_split(housing.data, housing.target, test_size=0.2)
    x_tr, x_val, y_tr, y_val = train_test_split(x_tr_, y_tr_, test_size=0.1)

    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_tr)
    x_val = scaler.transform(x_val)
    x_ts = scaler.transform(x_ts)
    return {'train input':x_tr, 'train target':y_tr,
            'test input':x_ts, 'test target':y_ts,
            'val input':x_val, 'val target':y_val}

def build_seq_net(D, argepochs=20, nhidden=30):
    # D is a doctionary
    model = keras.models.Sequential([
        keras.layers.Dense(nhidden, activation="relu", input_shape=D["train input"].shape[1:]),
        keras.layers.Dense(1)
    ])

    model.compile(loss="mean_squared_error", optimizer="sgd", metrics=['mse'])
    hist = model.fit(D["train input"], D["train target"],
                     epochs=argepochs,
                     validation_data=(D["val input"],D["val target"]))
    mse_test = model.evaluate(D["test input"], D["test target"])
    return {"model":model, "history": hist, "test mse":mse_test}

def build_nonseq_net(D, argepochs = 20):
    input_ = keras.layers.Input(shape=D["train input"].shape[1:])
    hidden1 = keras.layers.Dense(30, activation="relu")(input_)
    hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
    concat = keras.layers.Concatenate()([input_, hidden2])
    output = keras.layers.Dense(1)(concat)
    model = keras.Model(inputs=[input_], outputs=[output])
    model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3), metrics=['mse'])
    hist = model.fit(D["train input"], D["train target"],
                     epochs=argepochs,
                     validation_data=(D["val input"], D["val target"]))
    mse_test = model.evaluate(D["test input"], D["test target"])
    return {"model": model, "history": hist, "test mse": mse_test}


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Hi")
    Ddict = obtain_scaled_data()
    mlp10 = build_seq_net(Ddict, argepochs=100, nhidden=10)
    mlp30 = build_seq_net(Ddict, argepochs=100, nhidden=30)
    mlp50 = build_seq_net(Ddict, argepochs=100, nhidden=50)
    widedeep = build_nonseq_net(Ddict, argepochs=500)

    plt.plot(widedeep['history'].history['mse'], label='#widedeep')
    plt.plot(mlp10['history'].history['mse'], label='#hidden 10')
    plt.plot(mlp30['history'].history['mse'], label='#hidden 30')
    plt.plot(mlp50['history'].history['mse'], label='#hidden 50')
    plt.legend()
    plt.savefig('all_mse_training.png')

    print("Wide deep: %.4f".format(widedeep['test mse']) )
    print("Wide deep: %.4f".format(mlp10['test mse']) )
    print("Wide deep: %.4f".format(mlp30['test mse']) )
    print("Wide deep: %.4f".format(mlp50['test mse']) )
