import tensorflow as tf
import numpy as np

#Number_of_rooms
xs = np.array([1, 2, 3, 4])
#Apartment_Rent_label
ys = np.array([10, 15, 20, 25])

#model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])

#compile
model.compile(optimizer="sgd", loss="mean_squared_error")
#fit
model.fit(xs, ys, epochs=1000)

#prediction
pred = model.predict([7])
print(f"Prediction : {pred[0][0]:.2f}")
