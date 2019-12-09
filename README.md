# Code of Multimodal Hypergraph Learning for Microblog Sentiment Prediction
This is the first code to implement Multimodal Hypergraph Learning (Multi-HGL)

## Brief Explanation
* run_CV_gridsearch.m: an entrance for transductive learning, evaluation and inference
* HG_learning.m: a core part for hypergraph learning (someone can use or advances it for other tasks)
* preprocess*.m: pre-processing codes for data (we were informed that the data was sensitive, so you can refer to the codes to pre-process your data)
* mPara.mStarExp, mPara.mLamda, mPara.mMu, and mPara.mProbSigmaWeight are four main hyper-parameters (Please refer to the paper), which can be obtained via findCVPara.m

## Citing Multi-HGL

If you find Multi-HGL code useful in your research, please consider citing:

    @inproceedings{chen2015multimodal,
      title={Multimodal hypergraph learning for microblog sentiment prediction},
      author={Chen, Fuhai and Gao, Yue and Cao, Donglin and Ji, Rongrong},
      booktitle={2015 IEEE International Conference on Multimedia and Expo (ICME)},
      pages={1--6},
      year={2015},
      organization={IEEE}
    }
