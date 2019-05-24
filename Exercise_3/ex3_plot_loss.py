import matplotlib.pyplot as plt
import numpy as np

files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
  print(f)
models=["pretrained_finetune.txt","pretrained.txt","baseline.txt"]

train_variablesList = dict()
test_variablesList = dict()
for model_name in models :
  train_variablesList[model_name]=list()
  test_variablesList[model_name]=list()

for model_name in models :
  with open(model_name, "rb") as fp:   # Unpickling
      train = pickle.load(fp)
      train_variablesList[model_name]=(train)
      valid=pickle.load(fp);
      test_variablesList[model_name]=(valid)

t = np.arange(1,len(train_variablesList[models[2]])+1,1)

#baseline Model
plt.plot(t,train_variablesList[models[2]])
plt.plot(t,test_variablesList[models[2]])

#Pretrained + fine tuned
plt.plot(t,train_variablesList[models[0]])
plt.plot(t,test_variablesList[models[0]])

# Only pretrained 
plt.plot(t,train_variablesList[models[1]])
plt.plot(t,test_variablesList[models[1]])
plt.title("Training and Validation loss vs Number of Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(['baseline Train', 'baseline Validation', 'Pretrained_Finetuned Train', 'Pretrained_Finetuned','Pretrained Train', 'Pretrained Validation'], loc='upper right')
plt.show()

