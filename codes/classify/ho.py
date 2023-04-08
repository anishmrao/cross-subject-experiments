#!/usr/bin/env python
# coding: utf-8
"""
Hold out classification analysis of BCI Comp IV-2a and Korea datasets
@author: Ravikiran Mane
"""
import numpy as np
import torch
import sys
import os
import time
import xlwt
import csv
import random
import math
import copy
import pickle
import gc

masterPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(masterPath, 'centralRepo'))
from eegDataset import eegDataset
from baseModel import baseModel
import networks
import transforms
from saveData import fetchData
import seaborn as sns
import matplotlib.pyplot as plt
# from prettytable import PrettyTable


# reporting settings
debug = False

def getLosoSplit(subs, val_sub=None, test_sub=None):
    subs_copy = copy.deepcopy(subs)
    if(not val_sub):
        val_sub_ind = random.choice(list(range(9)))
        val_sub = subs[val_sub_ind]

    if(not test_sub):
        test_sub_ind = random.choice(list(range(9)))
        while(test_sub_ind == val_sub_ind):
            test_sub_ind = random.choice(list(range(9)))
        test_sub = subs[test_sub_ind]

    train_subs = [x for x in subs if x not in test_sub and x not in val_sub]

    print("Created LOSO Split: Train subjects:", train_subs, "Val subject:", val_sub, "Test subject:", test_sub)
    del subs_copy
    gc.collect()
    return train_subs, val_sub, test_sub

def ho(datasetId = None, network = None, nGPU = None, subTorun=None):
    #%% Set the defaults use these to quickly run the network
    datasetId = datasetId or 0
    network = network or 'FBCNet'
    nGPU = nGPU or 0
    subTorun= subTorun or None
    selectiveSubs = False
    
    # decide which data to operate on:
    # datasetId ->  0:BCI-IV-2a data,    1: Korea data
    datasets = ['bci42a', 'korea']
    
    #%% Define all the model and training related options here.
    config = {}

    # Data load options:
    config['preloadData'] = False # whether to load the complete data in the memory

    # Random seed
    config['randSeed']  = 20230310
    
    # Network related details
    config['network'] = network
    config['batchSize'] = 16
    
    if datasetId == 1:
        config['modelArguments'] = {'nChan': 20, 'nTime': 1000, 'dropoutP': 0.5,
                                    'nBands':9, 'm' : 32, 'temporalLayer': 'LogVarLayer',
                                    'nClass': 2, 'doWeightNorm': True}
    elif datasetId == 0:
        config['modelArguments'] = {'nChan': 22, 'nTime': 1000, 'dropoutP': 0.5,
                                    'nBands':9, 'm' : 32, 'temporalLayer': 'LogVarLayer',
                                    'nClass': 4, 'doWeightNorm': True}
    
    # Training related details    
    config['modelTrainArguments'] = {'stopCondi':  {'c': {'Or': {'c1': {'MaxEpoch': {'maxEpochs': 1000, 'varName' : 'epoch'}},
                                                       'c2': {'NoDecrease': {'numEpochs' : 200, 'varName': 'valInacc'}} } }},
          'classes': [0,1], 'sampler' : 'RandomSampler', 'loadBestModel': True,
          'bestVarToCheck': 'valInacc', 'continueAfterEarlystop':False,'lr': 1e-3}
            
    if datasetId ==0:
        config['modelTrainArguments']['classes'] = [0,1,2,3] # 4 class data

    config['transformArguments'] = None

    # add some more run specific details.
    config['cv'] = 'trainTest'
    config['kFold'] = 1
    config['data'] = 'raw'
    config['subTorun'] = subTorun
    config['trainDataToUse'] = 1    # How much data to use for training
    config['validationSet'] = 0.2   # how much of the training data will be used a validation set

    # network initialization details:
    config['loadNetInitState'] = True
    config['pathNetInitState'] = config['network'] + '_'+ str(datasetId)

    #%% Define data path things here. Do it once and forget it!
    # Input data base folder:
    toolboxPath = os.path.dirname(masterPath)
    config['inDataPath'] = os.path.join(toolboxPath, 'data')
    
    # Input data datasetId folders
    if 'FBCNet' in config['network']:
        modeInFol = 'multiviewPython' # FBCNet uses multi-view data
    else:
        modeInFol = 'rawPython'

    # set final input location
    config['inDataPath'] = os.path.join(config['inDataPath'], datasets[datasetId], modeInFol)

    # Path to the input data labels file
    config['inLabelPath'] = os.path.join(config['inDataPath'], 'dataLabels.csv')

    # Output folder:
    # Lets store all the outputs of the given run in folder.
    config['outPath'] = os.path.join(toolboxPath, 'output')
    config['outPath'] = os.path.join(config['outPath'], datasets[datasetId], 'ses2Test')

    # Network initialization:
    config['pathNetInitState'] = os.path.join(masterPath, 'netInitModels', config['pathNetInitState']+'.pth')
    # check if the file exists else raise a flag
    config['netInitStateExists'] = os.path.isfile(config['pathNetInitState'])
    
    #%% Some functions that should be defined here

    def setRandom(seed):
        '''
        Set all the random initializations with a given seed

        '''
        # Set np
        np.random.seed(seed)

        # Set torch
        torch.manual_seed(seed)

        # Set cudnn
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def excelAddData(worksheet, startCell, data, isNpData = False):
        '''
            Write the given max 2D data to a given given worksheet starting from the start-cell.
            List will be treated as a row.
            List of list will be treated in a matrix format with inner list constituting a row.
            will return the modified worksheet which needs to be written to a file
            isNpData flag indicate whether the incoming data in the list is of np data-type
        '''
        #  Check the input type.
        if type(data) is not list:
            data = [[data]]
        elif type(data[0]) is not list:
            data = [data]
        else:
            data = data

        # write the data. starting from the given start cell.
        rowStart = startCell[0]
        colStart = startCell[1]

        for i, row in enumerate(data):
            for j, col in enumerate(row):
                if isNpData:
                    worksheet.write(rowStart+i, colStart+j, col.item())
                else:
                    worksheet.write(rowStart+i, colStart+j, col)

        return worksheet

    def dictToCsv(filePath, dictToWrite):
    	"""
    	Write a dictionary to a given csv file
    	"""
    	with open(filePath, 'w') as csv_file:
    		writer = csv.writer(csv_file)
    		for key, value in dictToWrite.items():
    			writer.writerow([key, value])

    def print_parameters(model):
        # table = PrettyTable(["Modules", "Parameters"])
        print(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            # table.add_row([name, params])
            print([name, params])
            total_params+=params
        # print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    #%% create output folder
    # based on current date and time -> always unique!
    randomFolder = str(time.strftime("%Y-%m-%d--%H-%M", time.localtime()))+ '-'+str(random.randint(1,1000))
    config['outPath'] = os.path.join(config['outPath'], randomFolder,'')
    # create the path
    if not os.path.exists(config['outPath']):
        os.makedirs(config['outPath'])
    print('Outputs will be saved in folder : ' + config['outPath'])
    
    log_file = os.path.join(config['outPath'], 'log.txt')
    def log_write(line):
        with open(log_file, 'a') as log:
            log.write(line + '\n')
    
    with open(log_file, 'w') as log:
            log.write("Starting training...")

    # Write the config dictionary
    dictToCsv(os.path.join(config['outPath'],'config.csv'), config)

    #%% Check and compose transforms
    if config['transformArguments'] is not None:
        if len(config['transformArguments']) >1 :
            transform = transforms.Compose([transforms.__dict__[key](**value) for key, value in config['transformArguments'].items()])
        else:
            transform = transforms.__dict__[list(config['transformArguments'].keys())[0]](**config['transformArguments'][list(config['transformArguments'].keys())[0]])
    else:
        transform = None

    #%% check and Load the data
    print(config)
    print('Data loading in progress')
    log_write('Data loading in progress')
    fetchData(os.path.dirname(config['inDataPath']), datasetId) # Make sure that all the required data is present!
    data = eegDataset(dataPath = config['inDataPath'], dataLabelsPath= config['inLabelPath'], preloadData = config['preloadData'], transform= transform)
    print('Data loading finished:', len(data))
    log_write('Data loading finished:' + str(len(data)))
    print("Data Labels:", data.labels[3])

    #%% Check and load the model
    #import networks
    if config['network'] in networks.__dict__.keys():
        network = networks.__dict__[config['network']]
    else:
        raise AssertionError('No network named '+ config['network'] + ' is not defined in the networks.py file')

    # Load the net and print trainable parameters:
    net = network(**config['modelArguments'])
    print('Trainable Parameters in the network are: ' + str(count_parameters(net)))
    log_write('Trainable Parameters in the network are: ' + str(count_parameters(net)))

    print_parameters(net)

    #%% check and load/save the the network initialization.
    if config['loadNetInitState']:
        if config['netInitStateExists']:
            netInitState = torch.load(config['pathNetInitState'])
        else:
            setRandom(config['randSeed'])
            net = network(**config['modelArguments'])
            netInitState = net.to('cpu').state_dict()
            torch.save(netInitState, config['pathNetInitState'])

   #%% Find all the subjects to run 
    subs = sorted(set([d[3] for d in data.labels]))
    nSub = len(subs)

    

    ## Set sub2run
    if selectiveSubs:
        config['subTorun'] = config['subTorun']
    else:
        if config['subTorun']:
            config['subTorun'] = list(range(config['subTorun'][0], config['subTorun'][1]))
        else:
            config['subTorun'] = list(range(nSub))

    print("subs, subTorun", subs, config['subTorun'])

    #%% Let the training begin
    trainResults = []
    valResults = []
    testResults = []

    def finetune_model(sub, config, data, model):
        print("Starting finetuning...")
        
        # if iSub not in config['subTorun']:
        #     return None, None
        
        start = time.time()
        
        # extract subject data
        subIdx = [i for i, x in enumerate(data.labels) if x[3] in sub]
        subData = copy.deepcopy(data)
        subData.createPartialDataset(subIdx, loadNonLoadedData = True)
        
        trainData = copy.deepcopy(subData)
        testData = copy.deepcopy(subData)
        
        # Isolate the train -> session 0 and test data-> session 1
        if len(subData.labels[0])>4:
            idxTrain = [i for i, x in enumerate(subData.labels) if x[4] == '0' ]
            idxTest = [i for i, x in enumerate(subData.labels) if x[4] == '1' ]
        else:
            raise ValueError("The data can not be divided based on the sessions")
        
        testData.createPartialDataset(idxTest)
        trainData.createPartialDataset(idxTrain)
        
        # extract the desired amount of train data: 
        trainData.createPartialDataset(list(range(0, math.ceil(len(trainData)*config['trainDataToUse']))))

        # isolate the train and validation set
        valData = copy.deepcopy(trainData)
        valData.createPartialDataset(list( range( 
            math.ceil(len(trainData)*(1-config['validationSet'])) , len(trainData))))
        trainData.createPartialDataset(list(range(0, math.ceil(len(trainData)*(1-config['validationSet'])))))

        outPathSub = os.path.join(config['outPath'], 'finetuned_sub' + sub)
        
        if(not os.path.exists(outPathSub)):
            os.makedirs(outPathSub)
        
        model.resultsSavePath = outPathSub
        model.train(trainData, valData, testData, **config['modelTrainArguments'])
        
        # extract the important results.
        trainResults.append([d['results']['trainBest'] for d in model.expDetails])
        valResults.append([d['results']['valBest'] for d in model.expDetails])
        testResults.append([d['results']['test'] for d in model.expDetails])
        
        # save the results
        results = {'train:' : trainResults[-1], 'val: ': valResults[-1], 'test': testResults[-1]}
        dictToCsv(os.path.join(outPathSub,'results.csv'), results)

        model_save_path = os.path.join(outPathSub, "network_state_dict.pth")
        torch.save(model.net.state_dict(), model_save_path)
        print("Saved model at", model_save_path)
        
        # Time taken
        print("Time taken = "+ str(time.time()-start))
        return model, results

    def train_model(train_subs, val_subs, test_sub, data, config, model=None, finetune=False, mdl=False, mdl_ratio=0.5, mainsub=None):
        print("Starting training...")
        log_write("In train_model for test sub " + str(test_sub))
        
        # iSubs = list(range(len(subs)))
        # if not all(iSub in config['subTorun']  for iSub in iSubs):
        #     return None, None
        
        start = time.time()
        
        # extract subject data
        if len(data.labels[0])>4:
            idxTrain = [i for i, x in enumerate(data.labels) if x[3] in train_subs and x[4] == '0' ]
            idxTest = [i for i, x in enumerate(data.labels) if x[3] in train_subs and x[4] == '1' ]
            idxVal = [i for i, x in enumerate(data.labels) if x[3] in val_sub and x[4] == '1' ]
        else:
            raise ValueError("The data can not be divided based on the sessions")
        
        if(mdl):
            log_write("Adding MDL data")
            subMdlIdx = [i for i, x in enumerate(data.labels) if x[3] in test_sub and x[4] == '0']
            mdl_split = int(len(subMdlIdx) * mdl_ratio)
            subMdlIdx = subMdlIdx[:mdl_split]
            idxTrain.extend(subMdlIdx)
        
        trainData = copy.deepcopy(data)
        trainData.createPartialDataset(idxTrain, loadNonLoadedData = True)
        log_write("Created trainData")
        testData = copy.deepcopy(data)
        testData.createPartialDataset(idxTest, loadNonLoadedData = True)
        log_write("Created testData")
        valData = copy.deepcopy(data)
        valData.createPartialDataset(idxVal, loadNonLoadedData = True)
        log_write("Created valData")
        
        
        # subData.createPartialDataset(subIdx, loadNonLoadedData = True)
        # log_write("Created MDL train data")
        
        # trainData = copy.deepcopy(subData)
        # log_write("After trainData deepcopy")
        # subIdxTest = [i for i, x in enumerate(data.labels) if x[3] in train_subs] # This had val subs test as well
        # subDataTest = copy.deepcopy(data)
        # log_write("After subDataTest deepcopy")
        # subDataTest.createPartialDataset(subIdxTest, loadNonLoadedData = True)
        # testData = copy.deepcopy(subDataTest)
        # log_write("After testData deepcopy")
        
        
        # # Isolate the train -> session 0 and test data-> session 1
        
        

        
        # testData.createPartialDataset(idxTest)
        # trainData.createPartialDataset(idxTrain)
        
        # # extract the desired amount of train data: 
        # trainData.createPartialDataset(list(range(0, math.ceil(len(trainData)*config['trainDataToUse']))))

        # # isolate the train and validation set
        # subIdxVal = [i for i, x in enumerate(data.labels) if x[3] in val_subs]
        # subDataVal = copy.deepcopy(data)
        # subDataVal.createPartialDataset(subIdxVal, loadNonLoadedData = True)
        # valData = copy.deepcopy(subDataVal)
        # if len(subDataVal.labels[0])>4:
        #     idxVal = [i for i, x in enumerate(subDataVal.labels) if x[4] == '1' ] # This was train ('0')
        # else:
        #     raise ValueError("The data can not be divided based on the sessions")
        
        # valData.createPartialDataset(idxVal)

        log_write("Finished creating train val test data")
        
        # Call the network for training
        if(not finetune):
            setRandom(config['randSeed'])
            net = network(**config['modelArguments'])
            net.load_state_dict(netInitState, strict=False)
            if(not mdl):
                outPathSub = os.path.join(config['outPath'], 'sub'+ str(test_sub))
            else:
                outPathSub = os.path.join(config['outPath'], 'sub_mdl_'+ str(test_sub))
            model = baseModel(net=net, resultsSavePath=outPathSub, batchSize= config['batchSize'], nGPU = nGPU)
        else:
            outPathSub = os.path.join(config['outPath'], 'sub'+ str(mainsub), 'finetuned_sub' + str(test_sub))
        
        if(not os.path.exists(outPathSub)):
            os.makedirs(outPathSub)
        
        log_write("Calling model.train")
        model.train(trainData, valData, testData, **config['modelTrainArguments'])
        log_write("Finished training")
        
        # extract the important results.
        trainResults.append([d['results']['trainBest'] for d in model.expDetails])
        valResults.append([d['results']['valBest'] for d in model.expDetails])
        testResults.append([d['results']['test'] for d in model.expDetails])
        
        # save the results
        results = {'train:' : trainResults[-1], 'val: ': valResults[-1], 'test': testResults[-1]}
        print("Results:", results)

        model_save_path = os.path.join(outPathSub, "network_state_dict.pth")
        torch.save(model.net.state_dict(), model_save_path)
        print("Saved model at", model_save_path)
        
        # Time taken
        print("Time taken = "+ str(time.time()-start))
        del trainData
        del testData
        del valData
        gc.collect()

        return model, results

    def test_model(test_sub, data, model):
        subIdxTest = [i for i, x in enumerate(data.labels) if x[3] in test_sub and x[4] == '1']
        testData = copy.deepcopy(data)
        testData.createPartialDataset(subIdxTest, loadNonLoadedData = True)
        # testData = copy.deepcopy(subDataTest)
        
        # # Isolate the train -> session 0 and test data-> session 1
        # if len(subDataTest.labels[0])>4:
        #     idxTest = [i for i, x in enumerate(subDataTest.labels) if x[4] == '1' ]
        # else:
        #     raise ValueError("The data can not be divided based on the sessions")
        
        # testData.createPartialDataset(idxTest)

        test_results = model.test(testData)

        del testData
        gc.collect()

        return test_results

    results_mat = np.zeros((9,9))
    all_test_results = dict()
    sub_accs = []
    finetune_accs = []
    finetune_results_dict = dict()
    test_result_accs = []
    all_results = dict()
    prev_test = None
    mdl = False

    subs_to_skip = [2,4,5,6,7,8,9]

    for iSub, sub in enumerate(subs):
        if(sub in subs_to_skip):
            continue
        train_subs, val_sub, test_sub = getLosoSplit(subs, val_sub=prev_test, test_sub=sub)
         
        model, results = train_model(train_subs, val_sub, test_sub, data, config, model=None, finetune=False, mdl=mdl, mainsub=None)
        print("Results for training on subjects:", train_subs, "val on", val_sub, "test on train subjects (subject specific results):", results)
        sub_accs.append(results['test'][0]['acc'])

        print("Testing on unseen subject data", test_sub)
        test_results = test_model(test_sub, data, model)
        all_test_results[sub] = test_results
        test_result_accs.append(test_results['acc'])
        print("Cross subject Test results:", test_results)

        results['cross_sub_test'] = test_results
        if(mdl):
            outPathSub = os.path.join(config['outPath'], 'sub_mdl_'+ str(test_sub))
        else:
            outPathSub = os.path.join(config['outPath'], 'sub'+ str(test_sub))
        dictToCsv(os.path.join(outPathSub,'results.csv'), results)

        # if(not model):
        #     continue
        
        # results_mat[iSub][iSub] = results['test'][0]['acc'] * 100
        all_results[sub] = results
        prev_test = sub

        if(mdl):
            with open(os.path.join(config['outPath'], "all_results.p"), 'wb') as handle:
                pickle.dump(all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
            with open(os.path.join(config['outPath'], "test_results.p"), 'wb') as handle:
                pickle.dump(test_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            print("Finished fold with test subject", sub)

            continue
        
        print("Freezing all but last layer weights...")
        for name, param in model.net.named_parameters():
            param.requires_grad = True if "lastLayer" in name else False

        print('Trainable Parameters in the network after freezing are: ' + str(count_parameters(model.net)))
        print_parameters(model.net)

        tuned_model, finetune_results = finetune_model(test_sub, config, data, model)
        finetune_results_dict[test_sub] = finetune_results
        finetune_accs.append(finetune_results['test'][0]['acc'])

        print("Results after finetuning on subject", test_sub, ":", finetune_results)
        
        with open(os.path.join(config['outPath'], "all_results.p"), 'wb') as handle:
            pickle.dump(all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(os.path.join(config['outPath'], "test_results.p"), 'wb') as handle:
            pickle.dump(test_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(os.path.join(config['outPath'], "finetune_results_dict.p"), 'wb') as handle:
            pickle.dump(finetune_results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print("Finished fold with test subject", sub)
        
    print("Saving Test Results...")
    with open(os.path.join(config['outPath'], "all_test_results.p"), 'wb') as handle:
        pickle.dump(all_test_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    test_result_accs = np.asarray(test_result_accs)
    with open(os.path.join(config['outPath'], "test_result_accs.p"), 'wb') as handle:
        pickle.dump(test_result_accs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    sub_accs = np.asarray(sub_accs)
    with open(os.path.join(config['outPath'], "sub_accs.p"), 'wb') as handle:
        pickle.dump(sub_accs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    finetune_accs = np.asarray(finetune_accs)
    with open(os.path.join(config['outPath'], "finetune_accs.p"), 'wb') as handle:
        pickle.dump(finetune_accs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Subject specific accuracies:", sub_accs)
    print("Mean of subject specific accuracies:", sub_accs.mean())
    print("Test accuracies:", test_result_accs)
    print("Mean test accuracy after LOSO training:", test_result_accs.mean())
    print("Finetuned test accuracies:", finetune_accs)
    print("Mean finetuned test accuracies:", finetune_accs.mean())


    # Save matrix heatmap
    # ax = sns.heatmap(results_mat, annot=True, xticklabels=list(range(1, 10)), yticklabels=list(range(1, 10)))

    # plt.title('Cross-subject test accuracies of EEGNet', fontsize = 16) # title with fontsize 20
    # plt.xlabel('Finetuned subject', fontsize = 8) # x-axis label with fontsize 15
    # plt.ylabel('Main Subject', fontsize = 8) # y-axis label with fontsize 15

    # plt.savefig(os.path.join(config['outPath'], 'test_acc_mat_' + network + '.png'))
    
    #%% Extract and write the results to excel file.

    # lets group the results for all the subjects using experiment.
    # the train, test and val accuracy and cm will be written

    # trainAcc = [[r['acc'] for r in result] for result in trainResults]
    # trainAcc = list(map(list, zip(*trainAcc)))
    # valAcc = [[r['acc'] for r in result] for result in valResults]
    # valAcc = list(map(list, zip(*valAcc)))
    # testAcc = [[r['acc'] for r in result] for result in testResults]
    # testAcc = list(map(list, zip(*testAcc)))

    # print("Results sequence is train, val , test")
    # print(trainAcc)
    # print(valAcc)
    # print(testAcc)

    # # append the confusion matrix
    # trainCm = [[r['cm'] for r in result] for result in trainResults]
    # trainCm = list(map(list, zip(*trainCm)))
    # trainCm = [np.concatenate(tuple([cm for cm in cms]), axis = 1) for cms in trainCm]

    # valCm = [[r['cm'] for r in result] for result in valResults]
    # valCm = list(map(list, zip(*valCm)))
    # valCm = [np.concatenate(tuple([cm for cm in cms]), axis = 1) for cms in valCm]

    # testCm = [[r['cm'] for r in result] for result in testResults]
    # testCm = list(map(list, zip(*testCm)))
    # testCm = [np.concatenate(tuple([cm for cm in cms]), axis = 1) for cms in testCm]

    # print(trainCm)
    # print(valCm)
    # print(testCm)
    #%% Excel writing
    # book = xlwt.Workbook(encoding="utf-8")
    # for i, res in enumerate(trainAcc):
    #     sheet1 = book.add_sheet('exp-'+str(i+1), cell_overwrite_ok=True)
    #     sheet1 = excelAddData(sheet1, [0,0], ['SubId', 'trainAcc', 'valAcc', 'testAcc'])
    #     sheet1 = excelAddData(sheet1, [1,0], [[sub] for sub in subs])
    #     sheet1 = excelAddData(sheet1, [1,1], [[acc] for acc in trainAcc[i]], isNpData= True)
    #     sheet1 = excelAddData(sheet1, [1,2], [[acc] for acc in valAcc[i]], isNpData= True)
    #     sheet1 = excelAddData(sheet1, [1,3], [[acc] for acc in testAcc[i]], isNpData= True)

    #     # write the cm
    #     for isub, sub in enumerate(subs):
    #         sheet1 = excelAddData(sheet1, [len(trainAcc[0])+5,0+isub*len( config['modelTrainArguments']['classes'])], sub)
    #     sheet1 = excelAddData(sheet1, [len(trainAcc[0])+6,0], ['train CM:'])
    #     sheet1 = excelAddData(sheet1, [len(trainAcc[0])+7,0], trainCm[i].tolist(), isNpData= False)
    #     sheet1 = excelAddData(sheet1, [len(trainAcc[0])+11,0], ['val CM:'])
    #     sheet1 = excelAddData(sheet1, [len(trainAcc[0])+12,0], valCm[i].tolist(), isNpData= False)
    #     sheet1 = excelAddData(sheet1, [len(trainAcc[0])+17,0], ['test CM:'])
    #     sheet1 = excelAddData(sheet1, [len(trainAcc[0])+18,0], testCm[i].tolist(), isNpData= False)

    # book.save(os.path.join(config['outPath'], 'results.xls'))

if __name__ == '__main__':
    arguments = sys.argv[1:]
    count = len(arguments)

    if count >0:
        datasetId = int(arguments[0])
    else:
        datasetId = None

    if count > 1:
        network = str(arguments[1])
    else:
        network = None

    if count >2:
        nGPU = int(arguments[2])
    else:
        nGPU = None

    if count >3:
        subTorun = [int(s) for s in str(arguments[3]).split(',')]

    else:
        subTorun = None
    
    ho(datasetId, network, nGPU, subTorun)

