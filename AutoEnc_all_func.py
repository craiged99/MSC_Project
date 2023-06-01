from tensorflow.keras.utils import plot_model
import tensorflow as tf

from tensorflow.python.keras.models import Model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.python.keras.layers import ZeroPadding1D, ZeroPadding2D, ZeroPadding3D, Input, Dense, Flatten, Reshape, Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Cropping2D
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from GetData import getData

import random
import pandas as pd
import h5py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

def generateDFs():

    back = pd.read_csv('Full_back.csv', index_col=0)
    
    A1 = getData('Sig1.h5')
    A2 = getData('Sig2.h5')
    A3 = getData('Sig3.h5')
    A4 = getData('Sig4.h5')
    
    scaler = MinMaxScaler(feature_range=(0, 1))

    scaler.fit(back)

    Back = scaler.transform(back)

    Sig1 = scaler.transform(A1)
    Sig2 = scaler.transform(A2)
    Sig3 = scaler.transform(A3)
    Sig4 = scaler.transform(A4)
    
    return Back,Sig1,Sig2,Sig3,Sig4



def Build_Data(no_of_background,back_sig_ratio,Back,Sig1,Sig2,Sig3,Sig4):

    back_red = Back[:no_of_background]
    sig1_red = Sig1[:int(no_of_background*(back_sig_ratio/4))]
    sig2_red = Sig2[:int(no_of_background*(back_sig_ratio/4))]
    sig3_red = Sig3[:int(no_of_background*(back_sig_ratio/4))]
    sig4_red = Sig4[:int(no_of_background*(back_sig_ratio/4))]
    
    full_dataset = np.row_stack([back_red,sig1_red,sig2_red,sig3_red,sig4_red])
    
    
    sig_back = [back_red,sig1_red,sig2_red,sig3_red,sig4_red]
    
    type_of_data = []
    
    for i in range(len(sig_back)):
        
        for j in range(len(sig_back[i])):
            type_of_data.append(i)
        
    X_train,X_test,Y_train,Y_test = train_test_split(full_dataset,type_of_data,test_size=0.5,shuffle=True,random_state=random.randint(1,10))
    
    X_test = np.vstack((X_train,X_test))
    Y_test = np.vstack((Y_train,Y_test)).reshape(len(X_test),)


    return X_test,Y_test,back_red


#nodes_of_layers = list

#ouputs visible enc (needed to build model)

def Build_Encoder(no_of_dense_layers,nodes_of_layers):

    visible_enc = Input(shape=(60,1))
    x = Reshape((10,6,1))(visible_enc)
    x = Conv2D(64, (3, 3), padding='same',activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (4, 4), padding='same',activation='relu')(x)
    x = Conv2D(8, (4, 4), padding='same',activation='relu')(x)
    #x = Conv2D(2, (4, 4), padding='same',activation='relu')(x)
    x = Flatten()(x)
    
    for i in range(no_of_dense_layers):
        
        x = Dense(nodes_of_layers[i],activation='relu')(x)
        
    
    model_enc = Model(inputs=visible_enc, outputs=x)
    model_enc.summary()
    
    return model_enc,visible_enc
    

def Build_Decoder(no_of_dense_layers,nodes_of_layers,model_enc):
    
    
    encoding_shape = model_enc.layers[-1].output_shape[1:]
    
    
    t = Input(shape=(np.prod(encoding_shape),))
    
    for i in range(no_of_dense_layers):
        
        if i == 0:
            
            y = Dense(nodes_of_layers[i],activation='relu')(t)
            
        else:
            
            y = Dense(nodes_of_layers[i],activation='relu')(y)
    
    y = Dense(30,activation='relu')(y)
    y = Dense(30,activation='relu')(y)
    y = Dense(150,activation='relu')(y)
    y = Reshape((5,3,10))(y)
    y = Conv2D(4, (4, 4), activation='relu', padding='same')(y)
    y = UpSampling2D((2, 2))(y)
    y = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(y)
    y = Reshape((60,1))(y)
    
    model_dec = Model(inputs=t, outputs=y)
    model_dec.summary()
    
    return model_dec


def Build_Model(visible_enc,model_enc,model_dec):
    
    i   = visible_enc
    cae = Model(i, model_dec(model_enc(i)))
    cae.summary()
    
    return cae


def Run_Model(model,back_red,epochs,batch_size,val_split):
    
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics='accuracy')
    history = model.fit(back_red[:len(back_red)].reshape(len(back_red),60,1), back_red[:len(back_red)].reshape(len(back_red),60,1), epochs=epochs, batch_size=batch_size,shuffle=True,validation_split=val_split)

    return model,history

#Below are a few functions that are used to test a model once it has been trained.

#This function works in conjunction with the 'Analyse_model function'. It takes an
# input of a 'cut-off' value, which cuts the generated mean-squared error values
# to a point, and says all values above show a signal event, while the values below
# show a SM event.


def Test_Model_MSE(cut_off,rms,Y_test):

    back_sig_pred = []
    accuracy_list = []
    predicted_as_signal = []
    predicted_as_back = []
    accuracy_list_2 = []
    
    
    for i in range(len(rms)):
        
        #rms = mean_squared error values (generated in the Analyse_model function)
        if rms[i] <= cut_off:
            predict = 0
            back_sig_pred.append(0)
            predicted_as_back.append(Y_test[i])
            
        else:
            predict = 1
            back_sig_pred.append(1)
            
            #If it predicts a signal, look at Y_test to see what it predicted
            predicted_as_signal.append(Y_test[i])
            
        #If it correctly predicts background    
        if predict == 0 and Y_test[i] == 0:
            accuracy_list.append(1)
            accuracy_list_2.append(0)
         
        #If it correctly predicts signal
        if predict !=0 and Y_test[i] > 0:
            accuracy_list.append(1)
            accuracy_list_2.append(1)
        
        #If it incorrectly predicts background
        if predict == 0 and Y_test[i] > 0:
            accuracy_list.append(0)
            accuracy_list_2.append(1.5)
            
        #If it incorrectly predicts singal
        if predict !=0 and Y_test[i] == 0:
            accuracy_list.append(0)
            accuracy_list_2.append(0.5)
    
    return accuracy_list, back_sig_pred,predicted_as_signal,predicted_as_back,accuracy_list_2



#This function is the exact same as above, however for binary_crossentropy
def Test_Model_BINARY(cut_off,avg_bce_values,Y_test):
        
    back_sig_pred = []
    accuracy_list = []
    predicted_as_signal = []
    predicted_as_back = []
    
    accuracy_list_2 = []
    
    for i in range(len(avg_bce_values)):
        
        
        #avg_bce_values = binary crossentropy values (generated in the Analyse_model function)
        if avg_bce_values[i] <= cut_off:
            predict = 0
            back_sig_pred.append(0)
            predicted_as_back.append(Y_test[i])
            
        else:
            predict = 1
            back_sig_pred.append(1)
            predicted_as_signal.append(Y_test[i])
        
            
        if predict == 0 and Y_test[i] == 0:
            accuracy_list.append(1)
            accuracy_list_2.append(0)
            
        if predict !=0 and Y_test[i] > 0:
            accuracy_list.append(1)
            accuracy_list_2.append(1)
            
        if predict == 0 and Y_test[i] > 0:
            accuracy_list.append(0)
            accuracy_list_2.append(1.5)
            
        if predict !=0 and Y_test[i] == 0:
            accuracy_list.append(0)
            accuracy_list_2.append(0.5)
        
    
    return accuracy_list, back_sig_pred,predicted_as_signal,predicted_as_back,accuracy_list_2




#This function does a number of things:
# - generate mean_squared error and binary crossentropy values for the decoded 
# values
# - Test a range of cut-off values for determining signal/background
# - For the best cut_off value, identify how successful the model was at 
# - identifying each signal
# - Plot the histograms of the mean_squared error and binary crossentropy values
def Analyse_model(model,X_test,Y_test):
    
    Sig_back_str = ['Background','Signal 1','Signal 2','Signal 3','Signal 4']
    
    pred = model.predict(X_test)
    
    #generate mean squared error values of decoded images against the original
    rms = []

    for i in range(len(pred)):
        rms.append(mean_squared_error(X_test[i],pred[i], squared=False))
        
    #generate binary crossentropy values of decoded images against the original
    avg_bce_values = []
    for i in range(len(pred)):
        x = difference(X_test[i].reshape(60,1),pred[i])
        avg_bce_values.append(x)
        
    avg_bce_values = [x for x in avg_bce_values if x <= 1000]
    avg_bce_values = [x for x in avg_bce_values if x >= -1000]
        
    cut_offs_mse = np.linspace(np.mean(rms)-np.mean(rms)/4,np.mean(rms)+np.mean(rms)/4,50)
    cut_offs_binary = np.linspace(np.mean(avg_bce_values)-np.mean(avg_bce_values)/4,np.mean(avg_bce_values)+np.mean(avg_bce_values)/4,50)

    accu_mse = []
    back_mse = []
    sig_mse = []

    for i in range(len(cut_offs_mse)):
        accuracy,bs_pred,pred_sig,pred_back,accuracy_test_2 = Test_Model_MSE(cut_offs_mse[i],rms,Y_test)
        
        try:
            back_correct = pd.Series(accuracy_test_2).value_counts().loc[0]
        except:
            back_correct = 0
            
        try:
            back_wrong = pd.Series(accuracy_test_2).value_counts().loc[0.5]
        except:
            back_wrong = 0
            
        try:
            sig_correct = pd.Series(accuracy_test_2).value_counts().loc[1]
        except:
            sig_correct = 0
            
        try:
            sig_wrong = pd.Series(accuracy_test_2).value_counts().loc[1.5]
        except:
            sig_wrong = 0

        accuracy_test = accuracy.count(1)/len(accuracy)
        
        back_mse.append((back_correct/(back_correct+back_wrong))*100)
        sig_mse.append((sig_correct/(sig_correct+sig_wrong))*100)
             
        accu_mse.append(accuracy_test)


    #find best cut_off
    best_cut_off_mse = cut_offs_mse[accu_mse.index(np.max(accu_mse))]

    accuracy_best_mse,bs_pred_best_mse,predicted_as_signal_mse,predicted_as_back_mse,accuracy_2 = Test_Model_MSE(best_cut_off_mse,rms,Y_test)

    accuracy_test_best_mse = accuracy_best_mse.count(1)/len(accuracy_best_mse)


    #Test different cut_off values for binary crossentropy
    accu = []
    back_b = []
    sig_b = []

    for i in range(len(cut_offs_binary)):
        accuracy,bs_pred,pred_sig,pred_back,accuracy_test_2 = Test_Model_BINARY(cut_offs_binary[i],avg_bce_values,Y_test)
        
        try:
            back_correct = pd.Series(accuracy_test_2).value_counts().loc[0]
        except:
            back_correct = 0
            
        try:
            back_wrong = pd.Series(accuracy_test_2).value_counts().loc[0.5]
        except:
            back_wrong = 0
            
        try:
            sig_correct = pd.Series(accuracy_test_2).value_counts().loc[1]
        except:
            sig_correct = 0
            
        try:
            sig_wrong = pd.Series(accuracy_test_2).value_counts().loc[1.5]
        except:
            sig_wrong = 0

        accuracy_test = accuracy.count(1)/len(accuracy)
        
        back_b.append((back_correct/(back_correct+back_wrong))*100)
        sig_b.append((sig_correct/(sig_correct+sig_wrong))*100)
        accu.append(accuracy_test)

    #find best cut_off
    best_cut_off = cut_offs_binary[accu.index(np.max(accu))]

    accuracy_best,bs_pred_best,predicted_as_signal,predicted_as_back,accuracy_test_2_best = Test_Model_BINARY(best_cut_off,avg_bce_values,Y_test)

    accuracy_test_best_binary = accuracy_best.count(1)/len(accuracy_best)

    #view success rate in identifying signals with mean squared error
    ratios_mse = []
    for i in range(5):

        #for each signal, identify how many were predicted to be a signal, and 
        # compare for many were actually in X_test
        if i  == 0:
            ratio = (pd.Series(predicted_as_back_mse).value_counts()[i])/pd.Series(Y_test[:len(pred)]).value_counts()[i]

            ratios_mse.append(ratio*100)
            
        else:
        
            try:
    
                ratio = (pd.Series(predicted_as_signal_mse).value_counts()[i])/pd.Series(Y_test[:len(pred)]).value_counts()[i]
    
                ratios_mse.append(ratio*100)
    
            except:
    
                ratios_mse.append(0)

    #view success rate in identifying signals with binary_crossentropy
    ratios_binary = []
    for i in range(5):

        #for each signal, identify how many were predicted to be a signal, and 
        # compare for many were actually in X_test
        
        if i  == 0:
            ratio = (pd.Series(predicted_as_back).value_counts()[i])/pd.Series(Y_test[:len(pred)]).value_counts()[i]

            ratios_binary.append(ratio*100)
            
        else:

            try:
    
                ratio = (pd.Series(predicted_as_signal).value_counts()[i])/pd.Series(Y_test[:len(pred)]).value_counts()[i]
    
                ratios_binary.append(ratio*100)
    
            except:
    
                ratios_binary.append(0)

    pos_idx_lst_mse = [ i for i, e in enumerate(bs_pred_best_mse) if (1 == e)]
    pos_idx_lst_bin = [ i for i, e in enumerate(bs_pred_best) if (1 == e)]
    join_pos = list(set(pos_idx_lst_mse + pos_idx_lst_bin))
    y_test_vals = [Y_test[i] for i in join_pos]

    ratios_total = []
    for i in range(5):

        #for each signal, identify how many were predicted to be a signal, and 
        # compare for many were actually in X_test
        
        if i == 0:
            ratio = (pd.Series(y_test_vals).value_counts()[i])/pd.Series(Y_test[:len(pred)]).value_counts()[i]

            ratios_total.append(100-(ratio*100))
            
        else:
            
    
            try:
                ratio = (pd.Series(y_test_vals).value_counts()[i])/pd.Series(Y_test[:len(pred)]).value_counts()[i]
    
                ratios_total.append(ratio*100)
    
            except:
    
                ratios_total.append(0)
        
    
    #Plot mean squared error and binary crossentropy histograms, along with their
    # respective 'best cut_off value'.
    
    back_places = [ i for i, e in enumerate(Y_test) if (0 == e)]
    sig_places = [ i for i, e in enumerate(Y_test) if (0 != e)]
    
    hist_back = [rms[i] for i in back_places]
    hist_sig = [rms[i] for i in sig_places]
    
    
    plt.hist(hist_back,bins=200,color='blue',alpha=0.5)
    plt.hist(hist_sig,bins=200,color='lightblue',alpha=0.5)
    plt.legend(['Background','Signal'])
    plt.axvline(x=best_cut_off_mse,color='crimson',linewidth=2)
    plt.title('MSE Best Cut-Off')
    plt.grid()
    plt.xlabel('Mean Squared Error Value')
    plt.ylabel('Events')
    plt.show()
    
    plt.plot(back_mse,sig_mse,color='blue',alpha=0.5)
    plt.grid()
    plt.xlabel('Background Accuracy (%)')
    plt.ylabel('Signal Accuracy (%)')
    plt.title('ROC Curve - MSE')
    plt.show()
    
    print('MSE Succes Ratess (Overall accuracy = '+ str(round(accuracy_test_best_mse*100,3))+'%):')
    print('')
    for i in range(len(Sig_back_str)):
        print(Sig_back_str[i] + ' = ' + str(round(ratios_mse[i],3))+'%')
    
    hist_back_bce = [avg_bce_values[i] for i in back_places]
    hist_sig_bce = [avg_bce_values[i] for i in sig_places]
    
    plt.hist(hist_back_bce,bins=200,color='darkgreen',alpha=0.5)
    plt.hist(hist_sig_bce,bins=200,color='lightgreen',alpha=0.5)
    plt.legend(['Background','Signal'])
    plt.title('Binary Crossentropy Best Cut-Off')
    plt.axvline(x=best_cut_off,color='crimson',linewidth=2)
    plt.grid()
    plt.xlabel('Binary Crossentropy Value')
    plt.ylabel('Events')
    plt.show()
    
    plt.plot(back_b,sig_b,color='darkgreen',alpha=0.5)
    plt.grid()
    plt.xlabel('Background Accuracy (%)')
    plt.ylabel('Signal Accuracy (%)')
    plt.title('ROC Curve - Binary Crossentropy')
    plt.show()
    
    
    print('')
    print('Binary Crossentropy Success Rates (Overall accuracy = '+ str(round(accuracy_test_best_binary*100,3))+'%):')
    print('')
    for i in range(len(Sig_back_str)):
        print(Sig_back_str[i] + ' = ' + str(round(ratios_binary[i],3))+'%')
        
    print('')
    print('Overall Success Rates:')
    print('')
    for i in range(len(Sig_back_str)):
        print(Sig_back_str[i] + ' = ' + str(round(ratios_total[i],3))+'%')
    

    return ratios_mse,ratios_binary,accuracy_test_best_mse,accuracy_test_best_binary,ratios_total,rms,avg_bce_values,back_mse,sig_mse,back_b,sig_b



#These are taken from an earlier checkpoint in order to generate binary crossentopy
# values.
def binary_crossentropy (img_in, img_out):
    assert img_in.shape == img_out.shape
    eps = np.finfo(float).eps
    img_out = np.clip(img_out, eps, 1. - eps)
    return - (img_in * np.log(img_out) + (1 - img_in) * np.log(1 - img_out))


def difference(img_in,img_out):
    
    #Take values of BCE between the original and predicted image
    P_BCE = binary_crossentropy(img_in,img_out)
    #Take average of all values
    avg_bce = np.average(P_BCE)
    return avg_bce


#string values for names used later


def Save_Model_Results(model,name):
    
    model.save('Models/'+name)
    
    
    
def Load_Model_and_Test(name):
    
    new_model = tf.keras.models.load_model('Models/'+name)
    
    ratios_mse,ratios_binary,accuracy_test_best_mse,accuracy_test_best_binary,ratios_total,rms,avg_bce_values = Analyse_model(new_model)
    
    
    
    return new_model
    





