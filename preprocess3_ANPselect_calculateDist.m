
clear all;

load datas/mImageTextFea.mat;
mDist = cell(3,1);%ÔÝÏÈ¿¼ÂÇone type of feature: Image

%**************only for image sentiment distance **************

%============================== ANP select ==============================
mImageTextFeaSel = cell(3,1);%ÔÝÏÈ¿¼ÂÇone type of feature: Image

mImageTextFeaSel{1,1} = mImageTextFea{1,1};

mImageTextFeaSel{2,1} = mImageTextFea{2,1};

mImageTextFeaSel{3,1} = zeros(size(mImageTextFea{3,1}));

selectThreshold = 0.8;
nImageNum = size(mImageTextFea{3,1},1);
nANPNum = size(mImageTextFea{3,1},2);
for i = 1:nImageNum
    ['ANP select ' num2str(i)]
    for j = 1:nANPNum
        if mImageTextFea{3,1}(i,j)>= selectThreshold
            mImageTextFeaSel{3,1}(i,j) = mImageTextFea{3,1}(i,j);
        else
            mImageTextFeaSel{3,1}(i,j) = 0;
        end
    end
end
save datas/mImageTextFeaSel.mat
%========== compute image sentiment distance based on ANPs ==============
%mImageNum = size(mImageTextFea{1,1},1);

nTextNum = size(mImageTextFea{1,1},1);
mDist{1,1} = zeros(nTextNum,nTextNum);
for iImg = 1:nTextNum
    ['compute Distance ' num2str(iImg)]
    for jImg = 1:nTextNum
        mDist{1,1}(iImg,jImg) = norm(mImageTextFeaSel{1,1}(iImg,:) - mImageTextFeaSel{1,1}(jImg,:));
    end
end

nFaceNum = size(mImageTextFea{2,1},1);
mDist{2,1} = zeros(nFaceNum,nFaceNum);
for iImg = 1:nFaceNum
    ['compute Distance ' num2str(iImg)]
    for jImg = 1:nFaceNum
        mDist{2,1}(iImg,jImg) = norm(mImageTextFeaSel{2,1}(iImg,:) - mImageTextFeaSel{2,1}(jImg,:));
    end
end

mDist{3,1} = zeros(nImageNum,nImageNum);
for iImg = 1:nImageNum
    ['compute Distance ' num2str(iImg)]
    for jImg = 1:nImageNum
        mDist{3,1}(iImg,jImg) = norm(mImageTextFeaSel{3,1}(iImg,:) - mImageTextFeaSel{3,1}(jImg,:));
    end
end
save datas/mDist.mat