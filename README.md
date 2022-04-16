# Example code for IDD Lite Dataset (http://idd.insaan.iiit.ac.in/dataset/download/)
This code is for helping newcomers understand semantic segmentation and the IDD Lite dataset. For using:

1. Clone this repository.
2. Please register, download and extract the IDD Lite dataset to a folder "idd20k_lite" in the same directory.
3. Run viewer.py to view the annotations overlaid on top of the corresponding image.
4. Run main.py to train a simple semantic segmentation model on the dataset.
5. [Coming Soon] Generate images for the test dataset to be uploaded to the leaderboard at: http://idd.insaan.iiit.ac.in/evaluation/submissions


Code is tested in Python 3.7.3. The following python packages needs to be installed: numpy, torch, matplotlib, scipy, glob, argparse.

http://cvit.iiit.ac.in/ncvpripg19/idd-challenge/
http://idd.insaan.iiit.ac.in/evaluation/submissions/