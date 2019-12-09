function [IniY gt distM nTest] = getTTSingleBigData(mDist,mPara,mTrainTestSplitBig)
%% getTTSmallData is to get the training and testing data from the small split data pool.
%% The input information is the number of testing round, i.e., iSmallTest
%% The output is the initialized labeled information IniY, the groundtruth data for all the data, and the used distance matrix distM

%% first get the id for the 10*10 results
    ix = mPara.iBigTest;    
    tmpDistM = mDist{mPara.iExp,mPara.iFea};% the feature matrix for all samples in the dataset
    
    % select the total training samples and the testing samples
    tmpPosTrain = mTrainTestSplitBig{ix,1};
    tmpPosTest = mTrainTestSplitBig{ix,2};
    tmpNegTrain = mTrainTestSplitBig{ix,3};
    tmpNegTest = mTrainTestSplitBig{ix,4};
    
    nPosTrain = length(tmpPosTrain); % number of postive training samples
    nNegTrain = length(tmpNegTrain); % number of negative training samples
    nPosTest = length(tmpPosTest); % number of postive testing samples
    nNegTest = length(tmpNegTest); % number of negative testing samples
    
    nTrain = nPosTrain + nNegTrain;
    nTest = nPosTest + nNegTest;
    nAll = nTrain + nTest;
    
    IniY = zeros(nAll,2);
    IniY(1:nPosTrain,1) = 1;% the positive training samples are given 1 in the first column
    IniY(nPosTrain+1:nTrain,2) = 1;% the negative training samples are given 1 in the second column

    %% gt for both training and testing
%     gt = zeros(nAll,1);
%     gt(1:nPosTrain,1) = 1;
%     gt(nPosTrain+1:nTrain,1) = 0;
%     gt(nTrain+1:nTrain+nPosTest,1) = 1;
%     gt(nTrain+nPosTest+1:nAll, 1) = 0;
    %% gt for testing only
    gt = zeros(nTest,1);
    gt(1:nPosTest,1) = 1;
    gt(nPosTest+1:nTest,1) = 2;
    
    tmpAllData = [tmpPosTrain;(tmpNegTrain+mPara.nAllPos);tmpPosTest;(tmpNegTest+mPara.nAllPos)];
    
    distM = zeros(nAll);
    for iImg1 = 1:nAll
        for iImg2 = 1:nAll
             pos1 = tmpAllData(iImg1);
             pos2 = tmpAllData(iImg2);
             if iImg1 == iImg2
                 distM(iImg1,iImg2) = 0;
             else
                 distM(iImg1,iImg2) = tmpDistM(pos1,pos2);
             end
        end
    end