from models.leaky_relu_model import LeakyReluModel
from agents.agent import Agent


model = LeakyReluModel()
agent = Agent(model)
print(model.model.summary())
agent.train()
model.save()
for i in range(0, 100):
    print(i, model.predict_num(i))



