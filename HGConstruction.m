function H = HGConstruction(vdistM,mPara)
%% this function is to construct the hypergraph
% input: IniY: the label information
%           vdistM: the distance matrix
%           mPara: the parameter
%           mPara.mLamda(mPara.iLamda)
%           mPara.mStarExp(mPara.iStarExp
% output: results4test: the classification results, 1 or 0
 
IS_ProH = mPara.IS_ProH;
nObject = size(vdistM{1,1},1);
mStarExp = mPara.mStarExp(mPara.iStarExp); % number of star expansion

mExpFea = mPara.mExpFea{mPara.iExp}; % the feature IDs used here
nExpFea = size(mExpFea,1); % the number of features used here
    
nEdge = nObject*nExpFea;

mProbSigmaWeight = mPara.mProbSigmaWeight(mPara.iProbSigmaWeight);
%% hypergraph construction
H =zeros(nObject,nEdge);

for iFeature = 1:nExpFea
    distM = vdistM{iFeature,1};
    aveDist = mean(mean(distM));
    for iObj = 1:nObject
        vDist = distM(iObj,:);
        %{
        if iObj==54
            aaa=1;
        end
        %}
        [values orders] = sort(vDist,'ascend');
        orders2 = orders(1:mStarExp);
        if isempty(find(orders2==iObj))
            values(mStarExp)=0;
            orders(mStarExp)=iObj;
        end
        for iLinked = 1:mStarExp
            if IS_ProH == 0 % if it is not pro H
                H(orders(iLinked),iObj+(iFeature-1)*nObject) = 1;
            else
                H(orders(iLinked),iObj+(iFeature-1)*nObject) = exp(-values(iLinked)^2/(mProbSigmaWeight*aveDist)^2);
            end
        end % end of iLinked
    end % end of iObj
end