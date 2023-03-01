# From Cell-Lines to Cancer Patients: Personalized Drug Synergy Prediction

---
## Abstract
Combination drug therapies are effective treatments for cancer. However, the genetic heterogeneity of the patients and exponentially large space of drug pairings pose significant challenges for finding the right combination for a specific patient. Current in silico prediction methods can be instrumental in reducing the vast number of candidate drug combinations. However, existing powerful methods are trained with cancer cell line gene expression data, which limits their applicability in clinical settings. While synergy measurements on cell lines models are available at large scale, patient-derived samples are too few to train a complex model. On the other hand, patient-specific single-drug response data are relatively more available. In this work, we propose a deep learning framework, Personalized DeepSynergy Predictor (PDSP), that enables us to use the patient-specific single drug response data for customizing patient drug synergy predictions. PDSP is first trained to learn synergy scores of drug pairs and their single drug presonses for a given cell line using drug structures and large scale cell line gene expression data. Then, the model is fine-tuned for patients with their patient gene expression data and associated single drug response measured on the patient ex vivo samples. In this study, we evaluate PDSP on data from three leukemia patients and observe that it improves the prediction accuracy by 27% compared to models trained on cancer cell line data

---

## Authors
Halil Ibrahim Kuru, A. Ercument Cicek, Oznur Tastan

Our paper is available at <a href="https://www.biorxiv.org/content/10.1101/2023.02.13.528276v2">**bioRxiv**</a>

---

## Instructions Manual

### Requirements
- Python 3.7
- Numpy 1.18.1 
- Scipy 1.4.1
- Pandas 1.0.1
- Tensorflow 2.1.0
- Tensorflow-gpu 2.1.0
- Scikit-Learn 0.22.1
- keras-metrics 1.1.0
- h5py 2.10.0
- cudnn 7.6.5 (for gpu support only)

### Data
Raw data of drug combinations are taken from <a href="https://drugcomb.fimm.fi/">**DrugComb**</a>

Drug chemical features are calculated by Chemopy

RMA normalized E-MTAB-3610 untrated cell line gene expression data is downloaded from <a href="https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources//Home_files/Extended%20Methods.html#8">**cancerrxgene**</a>

Patient synergy and sensitivity data is obtained from the Supplementary Files of He et al. <a href="https://aacrjournals.org/cancerres/article/78/9/2407/633083/Patient-Customized-Drug-Combination-Prediction-and">**He et al.**</a>

You can download preprocessed data from <a href="https://drive.google.com/open?id=1_x2AKkMgg9gX19fVDEujpmjeMoyjTLJZ">**link**</a>, extract all files into `data/`

### Training
```shell
$ python main.py --saved-model-name pdsp.h5 --train-test-mode 1
```

### Testing with pretrained Model
Download pretrained weights from <a href="https://drive.google.com/open?id=172zyZnJtdONf9jyArbY0vyLJMxYeQndF">**link**</a>

```shell
$ python main.py --saved-model-name best_model.h5 --train-test-mode 0
```

---

## References
- Zagidullin, B., Aldahdooh, J., Zheng, S., Wang, W., Wang, Y., Saad, J., ... & Tang, J. (2019). DrugComb: an integrative cancer drug combination data portal. Nucleic acids research, 47(W1), W43-W51.
- CCao, D. S., Xu, Q. S., Hu, Q. N., & Liang, Y. Z. (2013). ChemoPy: freely available python package for computational biology and chemoinformatics. Bioinformatics, 29(8), 1092-1094.
- [dataset] Francesco Iorio (2015). Transcriptional Profiling of 1,000 human cancer cell lines, arrayexpress-repository, V1. https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-3610.
- [patient dataset] He, L., Tang, J., Andersson, E.I., Timonen, S., Koschmieder, S., Wennerberg, K., Mustjoki, S., Aittokallio, T.: Patient-customized drug combination prediction and testing for t-cell prolymphocytic leukemia patients. Cancer research 78(9), 2407–2418 (2018)


## License

- **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**
