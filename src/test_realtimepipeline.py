"""
This is the main file for testing realtime performance.

Copyright 2015 Markus Oberweger, ICG,
Graz University of Technology <oberweger@icg.tugraz.at>

This file is part of DeepPrior.

DeepPrior is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

DeepPrior is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with DeepPrior.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import numpy
from data.dataset import NYUDataset, ICVLDataset
from net.poseregnet import PoseRegNetParams, PoseRegNet
from net.resnet import ResNetParams, ResNet
from net.scalenet import ScaleNetParams, ScaleNet
from util.realtimehandposepipeline import RealtimeHandposePipeline
from data.importers import ICVLImporter, NYUImporter, MSRA15Importer
from util.cameradevice import Kinect2Device, FileDevice


__author__ = "Markus Oberweger <oberweger@icg.tugraz.at>"
__copyright__ = "Copyright 2015, ICG, Graz University of Technology, Austria"
__credits__ = ["Markus Oberweger"]
__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Markus Oberweger"
__email__ = "oberweger@icg.tugraz.at"
__status__ = "Development"

if __name__ == '__main__':
    rng = numpy.random.RandomState(23455)

    # di = MSRA15Importer('../data/MSRA15/')
    # Seq2 = di.loadSequence('test_seq_1')
    # testSeqs = [Seq2]

    # di = ICVLImporter('/media/alex/Network Bac/Data Sets/ICVL/')
    # Seq2 = di.loadSequence('test_seq_1')
    # testSeqs = [Seq2]

    # di = NYUImporter('../data/NYU/')
    # Seq2 = di.loadSequence('test_1')
    # testSeqs = [Seq2]

    # load trained network
    # poseNetParams = ResNetParams(type=2, nChan=1, wIn=128, hIn=128, batchSize=1, numJoints=16, nDims=3)
    poseNetParams = PoseRegNetParams(type=11, nChan=1, wIn=128, hIn=128, batchSize=1, numJoints=14, nDims=3)
    poseNetParams.loadFile = "/home/alex/Downloads/deep-prior-weights/NYU_network_prior.pkl"
    comrefNetParams = ScaleNetParams(type=1, nChan=1, wIn=128, hIn=128, batchSize=1, resizeFactor=2, numJoints=1, nDims=3)
    comrefNetParams.loadFile = "/home/alex/Downloads/deep-prior-weights/net_NYU_COM.pkl"
    # config = {'fx': 588., 'fy': 587., 'cube': (300, 300, 300)}
    # config = {'fx': 241.42, 'fy': 241.42, 'cube': (250, 250, 250)}
    config = {'fx': 368., 'fy': 368., 'cube': (300, 300, 300)} # Kinect V2
    # config = {'fx': 224.5, 'fy': 230.5, 'cube': (300, 300, 300)}  # Creative Gesture Camera
    di = NYUImporter("./capture/")
    di.fx = config['fx']
    di.fy = config['fy']
    di.ux = 261.
    di.uy = 202.
    rtp = RealtimeHandposePipeline(poseNetParams, config, di, verbose=False, comrefNet=comrefNetParams)

    # use filenames
    # filenames = []
    # for i in testSeqs[0].data:
    #     filenames.append(i.fileName)
    # dev = FileDevice(filenames, di)

    # # use depth camera
    # dev = CreativeCameraDevice(mirror=True)
    dev = Kinect2Device(mirror=True)
    rtp.processVideo(dev)
