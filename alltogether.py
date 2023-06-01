from AutoEnc_all_func import generateDFs, Build_Data, Build_Encoder, Build_Decoder,Build_Model,Run_Model,Test_Model_MSE, Test_Model_BINARY, Analyse_model, Save_Model_Results, Load_Model_and_Test
import pandas as pd
import numpy as np

no_background = 20000
back_sig_ratio = 1.0

no_of_enc_layers = 3
nodes_in_enc = [150,30,30]

no_of_dec_layers = 3
nodes_in_dec = [30,30,150]

no_of_epochs = 7
batch_size = 500
validation_split = 0.2


#############################################################################

#Constants

#Get dataframes for each dataset
#Back,Sig1,Sig2,Sig3,Sig4 = generateDFs()

#Get datafranes ready to be used
X_test,Y_test,back_red = Build_Data(no_background, back_sig_ratio,Back,Sig1,Sig2,Sig3,Sig4)


column_names = ['Back','Sig1','Sig2','Sig3','Sig4','MSE-Back','MSE-Sig1','MSE-Sig2','MSE-Sig3','MSE-Sig4',
               'Bin-Back','Bin-Sig1','Bin-Sig2','Bin-Sig3','Bin-Sig4']

Varibables = ['No_Back','Back_Sig_Ratio','No_enc_dense','Enc_nodes','No_dec_dense','Dec_nodes','Epochs',
              'Batch_size','Val_split']

ratio_df = pd.DataFrame(columns=column_names)

save_variables = pd.DataFrame(columns = Varibables)


############################################################################

x = [100,150,500]


for i in range(len(x)):
    
    nodes_in_enc = [x[i],30,30]

    
    
    model_enc,visible_enc = Build_Encoder(no_of_enc_layers,nodes_in_enc)
    
    model_dec = Build_Decoder(no_of_dec_layers, nodes_in_dec,model_enc)
    
    ae = Build_Model(visible_enc, model_enc, model_dec)
    
    Run_Model(ae, back_red, no_of_epochs ,batch_size, validation_split)
    
    
    
    ratios_mse,ratios_binary,accuracy_test_best_mse,accuracy_test_best_binary,ratios_total,rms,avg_bce_values,back_mse,sig_mse,back_b,sig_b = Analyse_model(ae,X_test,Y_test)
    
    Save_Model_Results(ae, 'Model'+str(i))
    
    
    Ratios = np.hstack([ratios_total,ratios_mse,ratios_binary])
    
    dic = {}
    
    for j in range(len(Ratios)):
        dic[column_names[j]] = Ratios[j]
    
    ratio_df = ratio_df.append(dic,ignore_index=True)
    
    
    df_mse = pd.DataFrame(np.vstack([back_mse,sig_mse]),index=('Background','Signal'))
    df_b = pd.DataFrame(np.vstack([back_b,sig_b]),index=('Background','Signal'))
    
    df_mse.to_csv('Models/ROCs/MSE'+str(i)+'.csv')
    df_b.to_csv('Models/ROCs/Binary'+str(i)+'.csv')
    
    
    variab = {'No_Back':no_background,'Back_Sig_Ratio':back_sig_ratio,'No_enc_dense':no_of_enc_layers,
              'Enc_nodes':nodes_in_enc,'No_dec_dense':no_of_dec_layers,'Dec_nodes':nodes_in_dec,
              'Epochs':no_of_epochs,'Batch_size':batch_size,'Val_split':validation_split}


    save_variables = save_variables.append(variab,ignore_index=True)
    
    
ratio_df.to_csv('Models/Ratios.csv')
save_variables.to_csv('Models/Model_Details.csv')



    