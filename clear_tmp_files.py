# This script will clear middle temp files in dataset
import os
import glob

code_dir = os.getcwd()+'/'
if __name__ == '__main__':
    file_list = glob.glob(code_dir+"data/*/CL.nii.gz")
    file_list.extend(glob.glob(code_dir+"data/*/CR.nii.gz"))
    file_list.extend(glob.glob(code_dir+"data/*/HT.nii.gz"))
    file_list.extend(glob.glob(code_dir+"data/*/TP.nii.gz"))
    for f in file_list:
        os.remove(f)
        print('remove', f)
    print('all temp files are removed..')

