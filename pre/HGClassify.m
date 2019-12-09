function     [Perf tmpBad] =  HGClassify(mPara,BestPara)
%% Perf is a 1 time 6 vector
mDist = mPara.mDist;
mTrainTestSplitBig = mPara.mTrainTestSplitBig;

mPara.iProbSigmaWeight = BestPara(1,1);
mPara.iFea = BestPara(1,2);
mPara.iStarExp = BestPara(1,3);
mPara.iLamda = BestPara(1,4);
mPara.iMu = BestPara(1,5);

[IniY gt vdistM nTest mPara] = getTTBigData(mDist,mPara,mTrainTestSplitBig);% get the required data for the current learning procedure
H = HGConstruction(vdistM,mPara);
[relMatrix mPara] = HG_learning(H,IniY,mPara);
[tmpPerf tmpBad] = evaluate(relMatrix,gt,nTest,mPara);

Perf = tmpPerf(:);