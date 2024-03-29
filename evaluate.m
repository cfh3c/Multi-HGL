function [tmpPerf bad] = evaluate(relMatrix,gt,nTest,mPara)
%% this function is to evaluate the results for the relevance matrix relMatrix and gt, where nTest is the number of testing samples
% input: relMatrix: the relevance matrix generated by HG learning
%           gt: the ground truth of all samples
%           nTest: the number of testing samples, which are listed as the
%           bottom of the results.
% output: tmpPerf: the counting of the results

nObject = size(relMatrix,1);


allresults = zeros(nObject,1);
for iObj = 1:nObject
    if relMatrix(iObj, 1) > relMatrix(iObj, 2)
        allresults(iObj, 1) = 1;
    else
        allresults(iObj,1) = 2;
    end
end

results4test = allresults(nObject-nTest+1:nObject,1);

%% count the experimental results
%                  pos_detected   neg_detected
%   pos_gt              a                     b
%   neg_gt              c                     d
tmpPerf = zeros(2);

bad(1,1) = 0;
bad(2,1) = 0;

bad1 = 0;
bad2 = 0;
nList = length(mPara.TrueList);
for iObj = 1:nTest
    if  gt(iObj) == 1 &&   results4test(iObj) == 2
        bad1 = bad1+1;
        bad(1,bad1) = mPara.TrueList(nList-nTest+iObj);
    elseif gt(iObj) == 2 &&   results4test(iObj) == 1
        bad2 = bad2+1;
        bad(2,bad2) = mPara.TrueList(nList-nTest+iObj);
    end
    
    tmpPerf(gt(iObj),results4test(iObj)) = tmpPerf(gt(iObj),results4test(iObj))+1;
end