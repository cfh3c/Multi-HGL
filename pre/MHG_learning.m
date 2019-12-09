function f = MHG_learning(vH,IniY,mPara)
%% conduct hypergraph learning in H

nHG = size(vH,1);

[nObject nEdge] = size(vH{1,1});

IsWeight = mPara.IsWeight;

% feaID = mExpFea(iFeature);% find the feature id for the iFeature th one
% mStarExp = mPara.optPara(feaID,1);%get the optimal mStarExp for feaID
% lamda = mPara.mLamda(mPara.iLamda); % the parameter in SSL, i.e., lamda

nIter = 20;

    
lamda = 1000;

mu = mPara.mMu(mPara.iMu);
IsWeight = mPara.IsWeight;

W = ones(nHG,1)/nHG;
%% learning on the hypergraph
for iHG = 1:nHG   
    H = vH{iHG,1};
    %% DV DE INVDE calculation

    DV = eye(nObject);
    for iObject = 1:nObject
       DV(iObject,iObject) = sum(H(iObject,:)); 
    end

    DE = eye(nEdge);
    for iEdge = 1:nEdge
        DE(iEdge,iEdge)=sum(H(:,iEdge));
    end

    DV2 = DV^(-0.5);
    INVDE = inv(DE);

    ThetaAll{iHG,1} = DV2*H*INVDE*H'*DV2;
end


if IsWeight == 0 % if no weight learnign is required
    Theta = W(1,1)*ThetaAll{1,1};
    if nHG > 1
        for iHG = 2:nHG
            Theta = Theta + W(iHG,1)*ThetaAll{iHG,1};
        end
    end
    eta = 1/(1+lamda);
    L2 = eye(nObject)-eta*Theta;
    f = (lamda/(1+lamda))*inv(L2) * IniY;  

elseif IsWeight == 1% learn the optimal multiple HG fusion
    
    flag = 1;
    for iteration = 1:nIter
        oldW = W;
        if flag == 0
        else
            Theta = W(1,1)*ThetaAll{1,1};
            if nHG > 1
                for iHG = 2:nHG
                    Theta = Theta + W(iHG,1)*ThetaAll{iHG,1};
                end
            end
            eta = 1/(1+lamda);
            L2 = eye(nObject)-eta*Theta;
            f = (lamda/(1+lamda))*inv(L2) * IniY;  


           %% update w_hg
            tmpSum = zeros(nHG,1);
            for iHG = 1:nHG
                tmpSum(iHG,1) = f(:,1)'*(eye(nObject) - ThetaAll{iHG,1})*f(:,1)+f(:,2)'*(eye(nObject)-ThetaAll{iHG,1})*f(:,2);
            end
            sum_tmp = sum(tmpSum);
            for iHG = 1:nHG
                W(iHG,1) = 1/nHG+sum_tmp/(2*nHG*mu)-tmpSum(iHG,1)/(2*mu);
            end
            disObj = norm(W-oldW);
            oldW = W;
            if disObj < 0.01
                flag = 0;
            end
        end
    end % end iteration

end