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

pop = 50
ep1 = 100
ep2 = 100


epochs = ep1 + ep2

num_exp = 30


def line_p(optimizer, lr, dname, p):
    dire = "results2-" + optimizer + "-" + str(lr) + "/" + dname + "/" + "[" + str(p) + "]"
    
    TrainErr_List, TestErr_List = [], []
                
    for i in range(1, num_exp):
                    
        TrainErr, TestErr = [], []
                    
        fname = dire + "/results-" + str(i) 
                    
        if not os.path.exists(fname):
            continue

        res = open(fname)
        errs = res.readlines()
        
        if errs == []:
            continue
                    
        for line in range(len(errs)):
            trainerr = float(errs[line].split()[2])
            TrainErr.append(trainerr)
            testerr = float(errs[line].split()[3])
            TestErr.append(testerr)
        
        if len(TestErr) > 180:
            TrainErr_List.append(TrainErr)
            TestErr_List.append(TestErr)
        

    TrainErr_fin, TestErr_fin = [np.median(x) for x in zip(*TrainErr_List)], [np.median(x) for x in zip(*TestErr_List)]
    
    return TestErr_fin
    
def line_gsgp(optimizer, lr, dname):
    p = (1, 0)
    dire = "results2-gd-0.01/" + dname + "/" + "[" + str(p) + "]"
    
    TrainErr_List, TestErr_List = [], []
                
    for i in range(1, num_exp):
                    
        TrainErr, TestErr = [], []
                    
        fname = dire + "/results-" + str(i) 
                    
        if not os.path.exists(fname):
            continue

        res = open(fname)
        errs = res.readlines()
        
        if errs == []:
            continue
                    
        for line in range(len(errs)):
            trainerr = float(errs[line].split()[2])
            TrainErr.append(trainerr)
            testerr = float(errs[line].split()[3])
            TestErr.append(testerr)
        
        if len(TestErr) > 180:
            TrainErr_List.append(TrainErr)
            TestErr_List.append(TestErr)
        

    TrainErr_fin, TestErr_fin = [np.median(x) for x in zip(*TrainErr_List)], [np.median(x) for x in zip(*TestErr_List)]
    
    return TestErr_fin

for optimizer in ["adam"]:
    for lr in [0.001, 0.01, 0.1]:
        for dname in ["yacht", "bioav", "slump", "toxicity", "ppb", "concrete", "airfoil", "parkinson"]:
            
            er_gsgp = line_gsgp(optimizer, lr, dname)
            
            er_1 = line_p(optimizer, lr, dname, (1, 1))            
            er_2 = line_p(optimizer, lr, dname, (1, 2))            
            er_5 = line_p(optimizer, lr, dname, (1, 5))            
            er_10 = line_p(optimizer, lr, dname, (1, 10))                        
            
            plt.plot(er_gsgp, label = "p2=0")
            plt.plot(er_1, label = "p2=1")
            plt.plot(er_2, label = "p2=2")
            plt.plot(er_5, label = "p2=5")
            plt.plot(er_10, label = "p2=10")         
            
            
            # plt.xlabel('Iteration')
            # plt.ylabel( 'Fitness')
            # plt.grid(axis="y", linestyle= '--', linewidth=0.5)
            plt.xlim(0, epochs)
            plt.legend()
                    
            dire_res = "loss/"
            dire_prop = dire_res + dname
            if not os.path.exists(dire_prop):
                os.mkdir(dire_prop)
                    
            plt.savefig(dire_prop + "/" + dname + "_" + optimizer + "_" + str(lr) + "test_loss.png")
            plt.savefig("l_paper/" + dname + "_" + optimizer + "_" + str(lr) + "test_loss.png")
            plt.close()






