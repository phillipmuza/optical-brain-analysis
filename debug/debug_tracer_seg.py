import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for tracer_segmentation
from tracer_segmentation import TracerSignalAnalyser
from skimage.io import imread

os.chdir(r"F:\tracer_uptake\p301s_mice\whole_brain_analysis_2\an2\fitc\debug")
img = imread(r"F:\tracer_uptake\p301s_mice\whole_brain_analysis_2\an2\fitc\debug\fitc.tif")

tracer_analyser = TracerSignalAnalyser(img)
tracer_analyser.process()