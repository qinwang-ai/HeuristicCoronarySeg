# HeuristicCoronarySeg
The paper "Heuristic Tubular Structure Discovery for Accurate Coronary Artery Segmentation with Limited CCTA Samples" under AAAI review
> Requirements
>
    pip install torch
    pip install scikit-image
    pip install numpy 
    pip install SimpleITK
    pip install numba
    pip install tdqm

> Dataset
> 
CORONARY-18 dataset: https://baidu.com

Please download and save to RotterDamCoronaryDataset.
Directory should like this:

> Inference

Generate initial tracking points using CGL module:
    
    python ct2tp.py  
    
Above command will generate "TP.nii.gz" and save in same directory as CT files.

Generate the initial centerline of CGL's initial segmentation:
    
    python tp2cl.py
    
Above command will generate "CL.nii.gz" and save in same directory as CT files.

Generate heart segmentation:

    python ct2ht.py
Above command will generate "HT.nii.gz" and save in same directory as CT files.

Heuristic Coronary Segmentation by BFS:

    python ct2cr.py 
Above command will generate coronary segmentation "CR.nii.gz" and print its Dice score.


