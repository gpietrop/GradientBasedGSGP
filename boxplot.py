import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os 

matplotlib.rc('font', **{'size': 8, 'weight': 'bold'})

sns.set(context='notebook',
        style='darkgrid',
        palette='muted',
        font='sans-serif',
        font_scale=1)

general_dire = "bp-paper"
for optimizer in ["gd", "adam"]:
    for lr in [0.001, 0.01, 0.1]:
        for dname in ["yacht", "bioav", "slump", "toxicity",  "airfoil", "concrete", "ppb", "parkinson"]:
        
            TrainErr_p, TestErr_p = [], []    
            
            p = (1, 0)
            dire = "results2-gd-0.01/" + dname + "/" + "[" + str(p) + "]"
            
            TrainErr, TestErr = [], []
            for i in range(1, 30):
                
                fname = dire + "/results-" + str(i)
                if not os.path.exists(fname):
                    continue
                res = open(fname)
               
                errs = res.readlines()
                    
                if errs == []:
                    continue

                x = errs[-1].split()

                TrainErr.append(float(x[2]))
                TestErr.append(float(x[3]))
                    
                
            TrainErr_p.append(TrainErr) 
            TestErr_p.append(TestErr) 
            
            for p in [(1, 1), (1, 2), (1, 5), (1, 10)]:
            
                
                dire = "results2-" + optimizer + "-" + str(lr) + "/" + dname + "/" + "[" + str(p) + "]"

                
                TrainErr, TestErr = [], []
                for i in range(1, 30):
                
                    fname = dire + "/results-" + str(i)
                    if not os.path.exists(fname):

                        continue
                    res = open(fname)
                    
                    errs = res.readlines()
                    

                    if errs == []:
                        continue
                    

                    x_1 = errs[-1].split()
                    x_2 = errs[-2].split()
                    x_3 = errs[-3].split()
                    x_4 = errs[-4].split()
                    x_5 = errs[-5].split()
                 

                    TrainErr.append(float(min(x_1[2], x_2[2], x_3[2], x_4[2], x_5[2])))
                    TestErr.append(float(min(x_1[3], x_2[3], x_3[3], x_4[3], x_5[3])))
                    
                
                TrainErr_p.append(TrainErr) 
                TestErr_p.append(TestErr) 
           
           
                 
            dire_res = "boxplot/" + dname  
            if not os.path.exists(dire_res):
                os.mkdir(dire_res)
               
            sns.utils.axlabel(xlabel = "fitness (train)", ylabel=None, fontsize=10)
            box_plot = sns.boxplot(data = TrainErr_p, 
                orient = "h",
                showfliers=False)

            plt.yticks(plt.yticks()[0], ['0', '1', '2', '5', '10'])
                    
            plt.title(dname + " Train " + str(optimizer) + " " + str(lr))    
            plt.savefig(dire_res + "/" + optimizer + "_" + str(lr) + "_Train.png")
            plt.close()

            
            # sns.utils.axlabel(xlabel="fitness (test)", ylabel=None, fontsize=10)
            box_plot = sns.boxplot(data = TestErr_p, 
                orient = "h",
                showfliers = False)

            plt.yticks(plt.yticks()[0], ['0', '1', '2', '5', '10'])
            plt.savefig(general_dire + "/" + dname + "_" + optimizer + "_" + str(lr) + "_Test.png")
            plt.title(dname + " Test " + str(optimizer) + " " + str(lr))    
            plt.savefig(dire_res + "/" + optimizer + "_" + str(lr) + "_Test.png")

            plt.close()        
                    
                    
