function BestPara = findCVPara(mPara)

mTrainTestSplitSmall = mPara.mTrainTestSplitSmall;

nProbSigmaWeight = length(mPara.mProbSigmaWeight);
nFea = length(mPara.mFea);
nStarExp = length(mPara.mStarExp);
nLamda = length(mPara.mLamda);
nMu = length(mPara.mMu);
nExp = length(mPara.mExpFea);
mDist = mPara.mDist;

iExp = mPara.iExp;
iData = mPara.iData;
iPerm = mPara.iPerm;

for iProbSigmaWeight = 1:nProbSigmaWeight
     mPara.iProbSigmaWeight = iProbSigmaWeight;
     for iFea = 1:nFea%nFea %iFea 1: bu   iFea 2: wu
        mPara.iFea = iFea;
        for iStarExp = 1:nStarExp % the star expansion in the hypergraph construction
            mPara.iStarExp = iStarExp;
            for iLamda = 1:nLamda% the parameter in the SSL
                mPara.iLamda = iLamda;
               for iMu = 1:nMu 
                    mPara.iMu = iMu;
                    xPerm =zeros(2);
                    nSmall = size(mTrainTestSplitSmall,2);
                    for iSmallTest = 1:nSmall
                            mPara.iSmallTest = iSmallTest;
                            [IniY gt vdistM nTest mPara] = getSmallData(mDist,mPara,mTrainTestSplitSmall);% get the required data for the current learning procedure
                            H = HGConstruction(vdistM,mPara);
                            [relMatrix mPara] = HG_learning(H,IniY,mPara);
                            [tmpPerf bad] = evaluate(relMatrix,gt,nTest, mPara);
                             xPerm = xPerm + tmpPerf;
                    end% end of iSmallPer
                    %tmpACC(iProbSigmaWeight,iFea,iStarExp,iLamda,iMu) = (xPerm(1,1)+xPerm(2,2))/sum(sum(xPerm));%accuracy: the percentage of all correctedly classified results
                    %tmpACC(iProbSigmaWeight,iFea,iStarExp,iLamda,iMu) = (xPerm(1,1)+10 * xPerm(2,2))/(xPerm(1,2)+xPerm(2,1)*10);
                    tmpACC(iProbSigmaWeight,iFea,iStarExp,iLamda,iMu) = (xPerm(1,1)+40 * xPerm(2,2))/(xPerm(1,2)+xPerm(2,1)*40);
                    acc = (xPerm(1,1)+xPerm(2,2))/sum(sum(xPerm));%accuracy: the percentage of all correctedly classified results
                    sen = xPerm(1,1)/sum(xPerm(1,:));%accuracy: the percentage of all correctedly classified positive results
                    spec = xPerm(2,2)/sum(xPerm(2,:));%accuracy: the percentage of all correctedly classified negative results
                    bac = 0.5*(sen+spec); 
                    ppv = xPerm(1,1)/sum(xPerm(:,1));
                    npv = xPerm(2,2)/sum(xPerm(:,2));
     
                    
                    ['iExp=' num2str(iExp) ' iData=' num2str(iData) ' iPerm=' num2str(iPerm) ' CV Param Finding' num2str(iProbSigmaWeight) ' ' num2str(iFea) ' ' num2str(iStarExp) ' ' num2str(iLamda) ' ' num2str(iMu) ' values ' num2str(mPara.mProbSigmaWeight(iProbSigmaWeight) )  ' ' num2str(mPara.mFea(iFea)) ' ' num2str(mPara.mStarExp(iStarExp)) ' ' num2str(mPara.mLamda(iLamda)) ' ' num2str(mPara.mMu(iMu))  ' acc = ' num2str(acc)  ' sen = ' num2str(sen)  ' spec = ' num2str(spec)  ' bac = ' num2str(bac)  ' ppv = ' num2str(ppv)  ' acc = ' num2str(npv)] 
               end%end of iMu
            end%end of iLamda
        end%end of iStarExp
     end% end of iFea
end % end of iProbSigmaWeight

%% find the best results
[amax maxind] = max(tmpACC(:));
[uProbSigmaWeight uFea uStarExp uLamda uMu] = ind2sub(size(tmpACC),maxind);
BestPara(1,1) = uProbSigmaWeight;
BestPara(1,2) = uFea;
BestPara(1,3) = uStarExp;
BestPara(1,4) = uLamda;
BestPara(1,5) = uMu;