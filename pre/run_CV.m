%%learn part
%% including the multiple running with small cross-validation, big cross-validation, evaluation, parameter tunning.


%  First, load required data
%  Second, select the data from the data pool
%  Run the program
%  Evaluation and counting
%  Finally run the program on the big pool
clear all;

%load mDistPCA;
load mDist;
mPara.mDist = mDist;
nData = 20;
nFea = 1;

%% parameters:
mPara.IsWeight = 1; % 0: do not learn weight   1: learn weight
mPara.mFea = [1];
mPara.mStarExp = [20  30 50 ]% 50]%[50]%[10 20 25 30 35 40 50]% current best: 30
mPara.mLamda  = [1e1 1e2 1e3 3e3 1e4]%[0.5*1e3 0.8*1e3 1e3 2e3 3e3 4e3] % current best: 3*1e3
mPara.mMu = [1e1 1e2 1e3]%[0.8*1e2 1e2 2e2 3e2 4e2]% current bet: 1e2
%mPara.mExpFea = {[1]; [2]; [3]; [4];[1;2];  [1;3]; [1;4]; [2;3]; [2;4]; [3;4]; [1;2;3]; [1;2;4]; [1;3;4]; [2,3,4]; [1;2;3;4]};
mPara.mExpFea = {[1]};
mPara.mProbSigmaWeight = [0.05]% 1]%[0.05 0.1 0.2 0.5 1] %current: 0.05
 
nAllPos = 463;                     nAllNeg = 133;
mPara.nAllPos = nAllPos;          mPara.nAllNeg = nAllNeg;
mPara.nIter = 20;

mPara.IS_ProH = 1;            
nExpFea = length(mPara.mExpFea);

%timeCount = zeros(nExp,nData);

%% first level: the used feature
for iExp = 1:nExpFea
    mPara.iExp = iExp;
    %% second level: the 10 data
    for iData = 1:nData
                mPara.iData = iData;
                filename = ['mTrainTestSplitBig' num2str(iData)];
                load(filename);
                mPara.mTrainTestSplitBig = mTrainTestSplitBig;
                filename = ['mTrainTestSplitSmall' num2str(iData)];
                load(filename);
                mPara.mTrainTestSplitSmall = mTrainTestSplitSmall;

                %% the thrid level: 10 cross validation
                tmpPerf = zeros(10,4);
                for iPerm = 1:10
                    mPara.iPerm = iPerm;
                    mPara.iBigTest = iPerm;
                    %% 1 get the small 10 cv for parameter selection
                    %% 2 select the  testing data and training data 
                    %% 3 use the selected parameter for the iPerm-th test
                    % small cv
                    % first calculate the overall performance for each
                    % parameter setting, and then select the best one 
                    BestPara = zeros(1,5);        
                    
                    BestPara = findCVPara(mPara);
                    %BestPara(1,1:5) = [1,1,1,1,1];
                    [tmpPerf(iPerm,:) tmpBad] = HGClassify(mPara,BestPara);
                    
                    mBad{iExp,iData}{iPerm,1} = tmpBad;
                    tmpacc = (tmpPerf(iPerm,1)+tmpPerf(iPerm,4))/sum(tmpPerf(iPerm,:));
                    tmpsen = tmpPerf(iPerm,1)/(tmpPerf(iPerm,1)+tmpPerf(iPerm,3));
                    tmpspec = tmpPerf(iPerm,4)/(tmpPerf(iPerm,2)+tmpPerf(iPerm,4));
                    tmpbac = 0.5*(tmpsen+tmpspec); 
                    tmpppv = tmpPerf(iPerm,1)/(tmpPerf(iPerm,1)+tmpPerf(iPerm,2));
                    tmpnpv =  tmpPerf(iPerm,4)/(tmpPerf(iPerm,3)+tmpPerf(iPerm,4));
                                            
                    ['iExp=' num2str(iExp) ' iData=' num2str(iData) ' iPerm=' num2str(iPerm) ' finish best parameter grid searching. The best parameter is ' num2str(BestPara(1))  ' ' num2str(BestPara(2)) ' ' num2str(BestPara(3)) ' ' num2str(BestPara(4)) ' ' num2str(BestPara(5))  ' values ' num2str(mPara.mProbSigmaWeight(BestPara(1)) )  ' ' num2str(mPara.mFea(BestPara(2))) ' ' num2str(mPara.mStarExp(BestPara(3))) ' ' num2str(mPara.mLamda(BestPara(4))) ' ' num2str(mPara.mMu(BestPara(5)))  ]
                    ['iExp=' num2str(iExp) ' iData=' num2str(iData) ' iPerm=' num2str(iPerm)  ' acc = ' num2str(tmpacc) ' sen = ' num2str(tmpsen) ' spec = ' num2str(tmpspec)  ' bac = ' num2str(tmpbac)  ' ppv = ' num2str(tmpppv)  ' npv = ' num2str(tmpnpv) ]
                    mBad{iExp,iData}{iPerm,1}
                    vBestPara{iExp,iData}(iPerm,:) = BestPara';
                    mACCAll{iExp,1}(iData,iPerm) = tmpacc;
                end
                
                sumTmpPerf = sum(tmpPerf);
                
                acc = (sumTmpPerf(1,1)+sumTmpPerf(1,4))/sum(sumTmpPerf);
                mASS{iExp,1}(iData,1)  = acc;
                sen = sumTmpPerf(1,1)/(sumTmpPerf(1,1)+sumTmpPerf(1,3));
                mASS{iExp,1}(iData,2) = sen;
                spec = sumTmpPerf(1,4)/(sumTmpPerf(1,2)+sumTmpPerf(1,4));
                mASS{iExp,1}(iData,3) = spec;
                bac  = 0.5*(mASS{iExp,1}(iData,2) +mASS{iExp,1}(iData,3) ); 
                mASS{iExp,1}(iData,4) = bac;
                ppv  = sumTmpPerf(1,1)/(sumTmpPerf(1,1)+sumTmpPerf(1,2));
                mASS{iExp,1}(iData,5) = ppv;
                npv  =  sumTmpPerf(1,4)/(sumTmpPerf(1,3)+sumTmpPerf(1,4));
                mASS{iExp,1}(iData,6) = npv;
                
                ['iExp=' num2str(iExp) ' iData=' num2str(iData) ' the acc = ' num2str(mASS{iExp,1}(iData,1)) ' the sen = ' num2str(mASS{iExp,1}(iData,2)) ' spec = ' num2str(mASS{iExp,1}(iData,3)) ' bac = ' num2str(mASS{iExp,1}(iData,4)) ' ppv = ' num2str(mASS{iExp,1}(iData,5)) ' nvp = ' num2str(mASS{iExp,1}(iData,6))] 
                
                mBadUnion{iExp, iData}{1,1} = mBad{iExp,iData}{iPerm,1}(1,:);
                mBadUnion{iExp, iData}{2,1} = mBad{iExp,iData}{iPerm,1}(2,:);                
                for iPerm = 2:10
                    mBadUnion{iExp, iData}{1,1} = [mBadUnion{iExp, iData}{1,1}, mBad{iExp,iData}{iPerm,1}(1,:)];
                    mBadUnion{iExp, iData}{2,1} = [mBadUnion{iExp, iData}{2,1}, mBad{iExp,iData}{iPerm,1}(2,:)];                   
                end
                'false 1'
                mBadUnion{iExp, iData}{1,1} = unique(sort(mBadUnion{iExp, iData}{1,1}));
                mBadUnion{iExp, iData}{1,1}
                'false 2'
                mBadUnion{iExp, iData}{2,1} = unique(sort(mBadUnion{iExp, iData}{2,1}));
                mBadUnion{iExp, iData}{2,1}
    end %end of iData
    [mMeanASS(iExp,:) mStdASS(iExp,:)] = getMeanStd(mASS{iExp,1});
    

end% end of iExpFea
