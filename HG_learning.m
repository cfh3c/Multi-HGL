function [f mPara] = HG_learning(H,IniY,mPara)
%% conduct hypergraph learning in H

[nObject nEdge] = size(H);
lamda = mPara.mLamda(mPara.iLamda); % the parameter in SSL, i.e., lamda
mu = mPara.mMu(mPara.iMu);
IsWeight = mPara.IsWeight;


W = eye(nEdge)/nEdge;

%% learning on the hypergraph
if IsWeight == 0 % if no weight learnign is required
    %% DV DE INVDE calculation
    DV = eye(nObject);
    for iObject = 1:nObject
       DV(iObject,iObject) = sum(H(iObject,:).*diag(W)'); 
    end

    DE = eye(nEdge);
    for iEdge = 1:nEdge
        DE(iEdge,iEdge)=sum(H(:,iEdge));
    end

    DV2 = DV^(-0.5);
    INVDE = inv(DE);

    T = DV2*H*INVDE*H'*DV2;
    eta = 1/(1+lamda);
    L2 = eye(nObject) - eta*T;
    f = inv(L2) * IniY;
elseif IsWeight == 1
    nIter = mPara.nIter;
    dif_obj = 0;           
    flag = 0;               
    fRecord{1,1} = IniY;   
    fRecord{2,1} = IniY;    
    fRecord{3,1} = IniY;
    tmpW3 = W;             
    tmpW2 = W;              
    tmpW1 = W;
    count = 0; % count >1, fail.  count =0 or 1: continue
    
    for iIter = 1:nIter+1
        if flag == 0
           iIter
           %allweightHG = W;
           %save results/allweightHG.mat
           %% DV DE INVDE calculation
           DV = eye(nObject);
           for iObject = 1:nObject
              DV(iObject,iObject) = sum(H(iObject,:).*diag(W)'); 
           end
           DE = eye(nEdge);
           for iEdge = 1:nEdge
               DE(iEdge,iEdge)=sum(H(:,iEdge)); 
           end
           DV2 = DV^(-0.5);
           INVDE = inv(DE);

           Theta = DV2*H*W*INVDE*H'*DV2;
           eta = 1/(1+lamda);
           L2 = eye(nObject)-eta*Theta;
           f = (lamda/(1+lamda))*inv(L2) * IniY;
           tmpF{iIter,1} = f;
%           fRecord{3,1} = fRecord{2,1}; fRecord{2,1} = fRecord{1,1};    fRecord{1,1} = f;                 

           %%calculate the objective function
           tmp_sum_w1 = 0;
           tmp_sum_w2 = 0;
           for iEdge  = 1:nEdge
               tmp_sum_w1 = tmp_sum_w1 + W(iEdge,iEdge);
               tmp_sum_w2 = tmp_sum_w2 + W(iEdge,iEdge)*W(iEdge,iEdge);
           end
           laplacian = f(:,1)' * (eye(nObject) - Theta) * f(:,1) + f(:,2)' * (eye(nObject) - Theta) * f(:,2);
           exploss =  lamda * (norm(f(:,1) - IniY(:,1)) + norm(f(:,2) - IniY(:,2)));
           m_obj(iIter,1) = laplacian + exploss + mu*tmp_sum_w2;
           if iIter > 1                   
               dif_obj = m_obj(iIter,1) - m_obj(iIter-1,1);
               objrecord(iIter,1) = dif_obj;
           end
           %% update w_hg

           if dif_obj > 0
               count = count+1;
           else
               count = 0;
           end
           
           if iIter>2&& m_obj(iIter,1) - m_obj(iIter-1,1)==0&&m_obj(iIter-1,1) - m_obj(iIter-2,1)==0
               count = count+1;
           end

           if iIter > nIter  || (count > 0)%||& (m_obj(iIter,1) - m_obj(1,1)<0)%%if dif is too small, stop iteration
               
                flag = 1;
                %iIter
                f = tmpF{iIter-1,1};
           else                      
               DV2H = DV2*H; % DV2H : Tau
               INVDEHDV2 = INVDE*H'*DV2; 
               DV2HINVDEHDV2 = DV2H*W*INVDEHDV2;
               if iIter ~= (nIter+1)
                   obj4AllEdges = (f(:,1)'*DV2HINVDEHDV2*f(:,1) + f(:,2)'*DV2HINVDEHDV2*f(:,2));

                    watch = zeros(nEdge,1);
                    tmpWold = W;
                    for iEdge = 1:nEdge
                        tmp_left = DV2H(:,iEdge);
                        tmp_right = INVDEHDV2(iEdge,:);
                        obj4OneEdge(iEdge,1) = (f(:,1)'*(tmp_left*W(iEdge,iEdge)*INVDE(iEdge,iEdge)*tmp_left')*f(:,1) + f(:,2)'*(tmp_left*W(iEdge,iEdge)*INVDE(iEdge,iEdge)*tmp_left')*f(:,2));
                        clear tmp_left tmp_right;
                         %['obj = ' num2str(dif_obj) ' 1/nEdge = ' num2str(1/nEdge)  ' obj4AllEdges/(2*mu*nEdge) = ' num2str(obj4AllEdges/(2*mu*nEdge)) ' obj4OneEdge = '  num2str(i) '  ' num2str(obj4OneEdge)]
                        watch(iEdge,1) = obj4OneEdge(iEdge,1);watch(iEdge,2) = obj4AllEdges - obj4OneEdge(iEdge,1)*nEdge;
                        watch(iEdge,3) = (obj4AllEdges - obj4OneEdge(iEdge,1)*nEdge)/(2*mu); watch(iEdge,4) = (obj4AllEdges - obj4OneEdge(iEdge,1)*nEdge)/(2*mu*nEdge);
                        A_watch = [dif_obj 1/nEdge obj4AllEdges/(2*nEdge) obj4OneEdge(iEdge,1)/2];%                                                                       
                    end
                    
                    obj4AllEdges = sum(obj4OneEdge);
                    for iEdge = 1:nEdge                  
                        tmpW(iEdge,iEdge) = 1/nEdge+(obj4OneEdge(iEdge,1)*nEdge - obj4AllEdges)/(2*mu*nEdge);%better-0  
                        watch(iEdge,5) = 1/nEdge+(obj4AllEdges - obj4OneEdge(iEdge,1)*nEdge)/(2*mu*nEdge);
                    end
                    
                    minValue = min(diag(tmpW));
                    if minValue > 0
                        W = tmpW;
                    else
                       diagW = diag(tmpW);
                       diagW = diagW - minValue + 1e-5;
                       diagW = diagW/(1 - nEdge*minValue + nEdge*1e-5);
                       W = diag(diagW);
                    end
                    
%                     minWeight = 0.00000001;
%                     meanObj = sum(obj4OneEdge)/nEdge;
%                     for iEdge = 1:nEdge
%                         vMu(iEdge,1) = (obj4OneEdge(iEdge,1)-meanObj)/(W(iEdge,iEdge)-minWeight);
%                     end
%                     
%                     mu = max(max(vMu)/2,1);
%                     for iEdge = 1:nEdge                  
%                         W(iEdge,iEdge) = 1/nEdge+(obj4AllEdges - obj4OneEdge(iEdge,1)*nEdge)/(2*mu*nEdge);%better-0  
%                         watch(iEdge,5) = 1/nEdge+(obj4AllEdges - obj4OneEdge(iEdge,1)*nEdge)/(2*mu*nEdge);
%                     end

                    watch(1,6) = obj4AllEdges;

                    %% make everything >= 0
%                     MUST_No_Minus = 0;
%                     if MUST_No_Minus == 0
%                     else
%                        vW = diag(W);
%                        posMinus = find(vW<0);
%                        sumMinus = -sum(vW(posMinus));
%                        if sumMinus == 0
%                        else
%                            vW(posMinus) = 0.000000001;
%                            posPlus = find(vW>=0);
%                            sumPlus = sum(vW(posPlus));
%                            nPlus = length(posPlus);
%                            ratio = sumMinus/sumPlus;
%                            for iPos = 1:nPlus
%                                vW(posPlus(iPos)) = vW(posPlus(iPos))*ratio;
%                            end                   
%                            W = diag(vW);
%                        end
%                     end

                    tmpW3 = tmpW2; tmpW2 = tmpW1;  tmpW1 = W;
                    clear DV2H INVDEHDV2 FTF L1 L2 L3 T;
                    %Record_W{i_class,iteration}{i_para_lamda,1} = W;             
               end
           end
       end
    end % end iteration
elseif IsWeight == 2% the method in Yu Jun's paper
  %% update W: each time update two values in W
    nIter = mPara.nIter;
    dif_obj = 0;           
    flag = 0;               
    fRecord{1,1} = IniY;   
    fRecord{2,1} = IniY;    
    fRecord{3,1} = IniY;
    tmpW3 = W;             
    tmpW2 = W;              
    tmpW1 = W;
    count = 0; % count >1, fail.  count =0 or 1: continue
    nIter = 20;
    for iIter = 1:nIter+1
        %% update F
        if flag == 0
           %% DV DE INVDE calculation
           DV = eye(nObject);
           for iObject = 1:nObject
              DV(iObject,iObject) = sum(H(iObject,:).*diag(W)'); 
           end
           DE = eye(nEdge);
           for iEdge = 1:nEdge
               DE(iEdge,iEdge)=sum(H(:,iEdge)); 
           end
           DV2 = DV^(-0.5);
           INVDE = inv(DE);

           Theta = DV2*H*W*INVDE*H'*DV2;
           eta = 1/(1+lamda);
           L2 = eye(nObject)-eta*Theta;
           f = (lamda/(1+lamda))*inv(L2) * IniY;
           tmpF{iIter,1} = f;

           %%calculate the objective function
           tmp_sum_w1 = 0;
           tmp_sum_w2 = 0;
           for iEdge  = 1:nEdge
               tmp_sum_w1 = tmp_sum_w1 + W(iEdge,iEdge);
               tmp_sum_w2 = tmp_sum_w2 + W(iEdge,iEdge)*W(iEdge,iEdge);
           end
           laplacian = f(:,1)' * (eye(nObject) - Theta) * f(:,1) + f(:,2)' * (eye(nObject) - Theta) * f(:,2);
           exploss =  lamda * (norm(f(:,1) - IniY(:,1)) + norm(f(:,2) - IniY(:,2)));
           m_obj(iIter,1) = laplacian + exploss + mu*tmp_sum_w2;
           if iIter > 1
               dif_obj = m_obj(iIter,1) - m_obj(iIter-1,1);
               objrecord(iIter,1) = dif_obj;
           end
           %% update w_hg

           if dif_obj > 0
               count = count+1;
           else
               count = 0;
           end

           if iIter > nIter  || (count > 0)%||& (m_obj(iIter,1) - m_obj(1,1)<0)%%if dif is too small, stop iteration
                flag = 1;
                %iIter
                f = tmpF{iIter-1,1};
           else                      
               DV2H = DV2*H; % DV2H : Tau
               INVDEHDV2 = INVDE*H'*DV2; 
               DV2HINVDEHDV2 = DV2H*W*INVDEHDV2;
               if iIter ~= (nIter+1)
                   obj4AllEdges = -(f(:,1)'*DV2HINVDEHDV2*f(:,1) + f(:,2)'*DV2HINVDEHDV2*f(:,2));

                    watch = zeros(nEdge,1);
                    tmpWold = W;
                    for iEdge = 1:nEdge
                        tmp_left = DV2H(:,iEdge);
                        tmp_right = INVDEHDV2(iEdge,:);
                        obj4OneEdge(iEdge,1) = -(f(:,1)'*tmp_left*W(iEdge,iEdge)*INVDE(iEdge,iEdge)*tmp_left'*f(:,1) + f(:,2)'*tmp_left*W(iEdge,iEdge)*INVDE(iEdge,iEdge)*tmp_left'*f(:,2));
                        clear tmp_left tmp_right;
                         %['obj = ' num2str(dif_obj) ' 1/nEdge = ' num2str(1/nEdge)  ' obj4AllEdges/(2*mu*nEdge) = ' num2str(obj4AllEdges/(2*mu*nEdge)) ' obj4OneEdge = '  num2str(i) '  ' num2str(obj4OneEdge)]
                        watch(iEdge,1) = obj4OneEdge(iEdge,1);watch(iEdge,2) = obj4AllEdges - obj4OneEdge(iEdge,1)*nEdge;
                        watch(iEdge,3) = (obj4AllEdges - obj4OneEdge(iEdge,1)*nEdge)/(2*mu); watch(iEdge,4) = (obj4AllEdges - obj4OneEdge(iEdge,1)*nEdge)/(2*mu*nEdge);
                        A_watch = [dif_obj 1/nEdge obj4AllEdges/(2*nEdge) obj4OneEdge(iEdge,1)/2];%                                                                       
                    end
                    nRand = 1;
                    for iRandEdge = 1:nRand 
                        [value] = rand(1,nEdge);
                        [sortedV sortedRank] = sort(value);
                        for iStart = 1:floor(nEdge/2)
                            edge1 = sortedRank((iStart-1)*2+1);
                            edge2 = sortedRank((iStart-1)*2+2);
                            w1 = W(edge1,edge1);
                            w2 = W(edge2,edge2);
                            s1 = obj4OneEdge(edge1,1);
                            s2 = obj4OneEdge(edge2,1);
                            para1 = 2*mu*(w1+w2)+(s2-s1);
                            para2 = 2*mu*(w1+w2)+(s1-s2);
% 
%                             if para1 < 0
%                                 W(edge1,edge1) = 0.0000001;
%                                 W(edge2,edge2) = w1+w2-0.0000001;
%                             elseif para2<0
%                                 W(edge2,edge2) = 0.0000001;
%                                 W(edge1,edge1) = w1+w2-0.0000001;
%                             else
%                                 W(edge1,edge1) = 0.5*(w1+w2)+(s2-s1)/(4*mu);
%                                 W(edge2,edge2) = w1+w2-W(edge1,edge1);                            
%                             end
                            if para1 < 0
                                W(edge1,edge1) = 0.1*w1;
                                W(edge2,edge2) = w1+w2-0.1*w1;
                            elseif para2<0
                                W(edge2,edge2) = 0.1*w2;
                                W(edge1,edge1) = w1+w2-0.1*w2;
                            else
                                W(edge1,edge1) = 0.5*(w1+w2)+(s2-s1)/(4*mu);
                                W(edge2,edge2) = w1+w2-W(edge1,edge1);                            
                            end
                        end                        
                    end% end of iRandEdge
                    %% update H and W
%                     
%                     diagW = diag(W);
%                     posNonZero = find(diagW>0);
%                     HNew = H(:,posNonZero);
%                     nEdge = length(posNonZero);
%                     WNew = diag(diagW(posNonZero));
%                    
%                     clear H W;
%                     H = HNew;
%                     W = WNew;
%                     
               end% end of if iIter~=(nIter+1)
           end% end of if iIter>nIter
       end
    end % end iteration
    
end