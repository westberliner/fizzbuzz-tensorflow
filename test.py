from models.leaky_relu_model import LeakyReluModel

model = LeakyReluModel()
model.load()
print(model.model.summary())

for i in range(0, 1000):
    print(model.predict_num(i))

model.save_to_json()

