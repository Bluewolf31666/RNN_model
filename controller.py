import os
from model.VGNET import Model as vgmodel
import preprocessing
import tensorflow
from model.Alex import AlexNet
###
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Model, Sequential
from keras import layers, optimizers, losses
import torch.optim as optim
import torch.nn as nn

class model_function():
  def VGGnet():
    train_gen,val_gen=preprocessing.data.VGGnet()
    targetSize = (150, 150)
    VGGNET_model=vgmodel.VGGnet_model(targetSize)

    epochs_count = 1
    History = VGGNET_model.fit(train_gen,
                        epochs = epochs_count,
                        verbose = 1,
                        validation_data = val_gen,
                      )
    return(History)

  #### alex

  def alex():
    model=AlexNet.Alex_model()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    cost = nn.CrossEntropyLoss()

    train_correct = []
    loss_record = []
    train_dataloader,test_dataloader = preprocessing.data.Alex()


    for epoch in range(5):

      for i, (images,labels) in enumerate(train_dataloader):

        optimizer.zero_grad()
    
        output = model(images)
        loss = cost(output, labels)
        loss.backward()
        optimizer.step()

      loss_record.append(loss.item())
      print(f'Epcoh [{epoch+1}/5], Loss: {loss.item():.4f}')
    return(output,loss_record)
  
  def LSTM():
    model.x
    model.fit(X_train, Y_train, epochs=10,verbose=1,validation_data=(X_test, Y_test))