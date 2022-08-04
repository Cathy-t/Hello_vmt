# Hello_vmt
Video-guided machine translation display platform

## Download the data
You can download the model and the feature of [MSVD-Turkish Dataset](https://hucvl.github.io/MSVD-Turkish/) from [here](https://drive.google.com/drive/folders/13XgukW41ryW3lVuN_yQEuuqe9nUuXSqX?usp=sharing). And the model was trained on [VATEX Dataset](https://eric-xw.github.io/vatex-website/download.html).  

## How to run the platform
* Download the processed video features in the [previous section](https://github.com/Cathy-t/Hello_vmt#download-the-data), or process the original video according to the corresponding paper.
* Modify 'TRAIN_VOCAB_EN'/'TRAIN_VOCAB_ZH'/'CHK_DIR'/'DATA_DIR' in the configuration file. 

>>>[configs_vret.yaml](https://github.com/Cathy-t/Hello_vmt/blob/main/src/configs_vret.yaml)  | VRET

>>>[configs_dear.yaml](https://github.com/Cathy-t/Hello_vmt/blob/main/src/configs_dear.yaml)  | DEAR

* run main.py and you can see the project like the one shown below .

<img src="https://github.com/Cathy-t/Hello_vmt/blob/main/vmt.gif" alt="show" />

## References
- https://github.com/kirbiyik/caption-it
- http://flask.pocoo.org/docs/1.0/deploying/
- https://github.com/Cathy-t/HELLO_image
- https://authors.elsevier.com/c/1eq-j3OAb95KRy
- https://d.wanfangdata.com.cn/patent/CN202110704424.6
- https://d.wanfangdata.com.cn/patent/CN202110395391.1

The implementation of the DEAR model is in the paper: 
[Video-guided machine translation via dual-level back-translation. Shiyu Chen ,Yawen Zeng ,Da Cao and Shaofei Lu. Knowledge-Based Systems 2022.](https://authors.elsevier.com/c/1eq-j3OAb95KRy)

```
@article{CHEN2022108598,
title = {Video-guided machine translation via dual-level back-translation},
journal = {Knowledge-Based Systems},
volume = {245},
pages = {108598},
year = {2022},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2022.108598},
url = {https://www.sciencedirect.com/science/article/pii/S0950705122002684},
author = {Shiyu Chen and Yawen Zeng and Da Cao and Shaofei Lu},
keywords = {Multiple modalities, Video-guided machine transaltion, Back-translation, Shared transformer},
abstract = {Video-guided machine translation aims to translate a source language description into a target language using the video information as additional spatio-temporal context. Existing methods focus on making full use of videos as auxiliary material, while ignoring the semantic consistency and reducibility between the source language and the target language. In addition, the visual concept is helpful for improving the alignment and translation of different languages but is rarely considered. Toward this end, we contribute a novel solution to thoroughly investigate the video-guided machine translation issue via dual-level back-translation. Specifically, we first exploit a sentence-level back-translation to obtain the coarse-grained semantics. Thereafter, a video concept-level back-translation module is proposed to explore the fine-grained semantic consistency and reducibility. Lastly, a multi-pattern joint learning approach is utilized to boost the translation performance. By experimenting on two real-world datasets, we demonstrate the effectiveness and rationality of our proposed solution.}
}
```

>The paper of the VRET model is In Press, Journal Pre-proof.
>[Vision talks: Visual relationship-enhanced transformer for video-guided machine translation. Shiyu Chen ,Yawen Zeng ,Da Cao and Shaofei Lu. Expert Systems with Applications 2022.](https://doi.org/10.1016/j.eswa.2022.118264)
```
@article{CHEN2022118264,
title = {Vision talks: Visual relationship-enhanced transformer for video-guided machine translation},
journal = {Expert Systems with Applications},
pages = {118264},
year = {2022},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2022.118264},
url = {https://www.sciencedirect.com/science/article/pii/S0957417422014051},
author = {Shiyu Chen and Yawen Zeng and Da Cao and Shaofei Lu},
keywords = {Machine translation, Visual relationship, Transformer, Graph convolutional network},
abstract = {Video-guided machine translation is a promising task which aims to translate a source language description into a target language utilizing the video information as supplementary context. The majority of existing work utilize the whole video as the auxiliary information to enhance the translation performance. However, visual information, as a heterogeneous modal with text, introduces noise instead. Toward this end, we propose a novel visual relationship-enhanced transformer by constructing a semantic-visual relational graph as a corss-modal bridge. Specifically, the visual information is regarded as the structured conceptual representation, which builds a bridge between two modalities. Thereafter, graph convolutional network is deployed to capture the relationship among visual semantics. In this way, a transformer with structured multi-modal fusion strategy is allowed to explore the correlations. Finally, the proposed framework is optimized under the scheme of Kullback–Leibler divergence with label smoothing. Extensive experiments demonstrate the rationality and effectiveness of our proposed method as compared to other state-of-the-art solutions.}
}
```


>研究生的主要研究成果都在这儿了，毕业快乐！不过该展示平台还是会慢慢更新的，毕竟是头秃的见证，不能白秃~
