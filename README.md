# HeuristicCoronarySeg
The paper "Heuristic Tubular Structure Discovery for Accurate Coronary Artery Segmentation with Limited CCTA Samples" under AAAI review
> Requirements
>
    pip install torch
    pip install scikit-image
    pip install numpy 
    pip install glob 
    pip install SimpleITK
    pip install numba
    pip install tdqm

> Dataset
> 
CORONARY-18 dataset: http://tmp.link/f/5f5a1199b2540

Please download and save to RotterDamCoronaryDataset.
Directory should like this:

[1]: /imgs/tree.png 

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

> Other

If you want to clear all temporary files of above process(note: this commend will delete predicted segmentation "CR.nii.gz" as well)
    
    python clear_tmp_files 

