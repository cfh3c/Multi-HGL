function [IniY gt vdistM nTest mPara] = getTTMultipleBigData(mDist,mPara,mTrainTestSplitBig)
%% getTTSmallData is to get the training and testing data from the big split data pool.
%% The input information is the number of testing round, i.e., iSmallTest
%% The output is the initialized labeled information IniY, the groundtruth data for all the data, and the used distance matrix distM

%% first get the id for the 10*10 results
    ix = mPara.iBigTest;   
    
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
    IniY(nPosTrain+1:nTrain,1) = -1;% 
    IniY(1:nPosTrain,2) = -1;% 
    IniY(nPosTrain+1:nTrain,2) = 1;% the negative training samples are given 1 in the second column

%    mSubW = diag([ones(1,nPosTrain)*mPara.mPosW ones(1,nNegTrain)*mPara.mNegW ones(1,nTest)/nTest]);
    mSubW = eye(nAll);
    mPara.mSubW = mSubW;
    
    
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
    
    mExpFea = mPara.mExpFea{mPara.iExp}; % the feature IDs used here
    nExpFea = size(mExpFea,1); % the number of features used here
    
    vdistM = cell(nExpFea,1);    
    for iFeature =1:nExpFea
        feaID = mExpFea(iFeature); % the used feature here
        tmpDistM = mDist{feaID,mPara.iFea};% the feature matrix for all samples in the dataset

        distM = zeros(nAll);
        for iImg1 = 1:nAll
            for iImg2 = 1:nAll
                 pos1 = tmpAllData(iImg1);
                 mPara.TrueList(iImg1,1) = pos1;
                 pos2 = tmpAllData(iImg2);
                 if iImg1 == iImg2
                     distM(iImg1,iImg2) = 0;
                 else
                     distM(iImg1,iImg2) = tmpDistM(pos1,pos2);
                 end
            end
        end
        vdistM{iFeature,1} = distM;
    end