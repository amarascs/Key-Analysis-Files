import numpy as np
import scipy
import uproot
import awkward as ak
import pandas as pd
import math
from numpy import random
from numpy.matlib import repmat
from scipy.optimize import curve_fit
from scipy.stats import skewnorm
from sklearn import mixture

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_pdf import PdfPages
import glob
import os
import importlib
import sys
import csv
import json
import datetime
import time as timelib

##########################################
## Functional forms for scipy.curve_fit ##
##########################################

# single gaussian - free amplitude, mean, standard deviation
def gaussian(x, *p):
    
    A, mu, sigma = p
    
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

# constrained double gaussian - assumes second peak comes from sampling first peak twice
# can be used to fit single- and double-PE peaks in SPE calibration
def gaussian_2(x, *p):
    
    A1, A2, mu_1, sigma_1 = p
    
    p1 = [A1, mu_1, sigma_1]
    p2 = [A2, 2*mu_1, np.sqrt(2)*sigma_1] # standard deviations add in quadrature
    
    f = gaussian(x, *p1) + gaussian(x, *p2)
    
    return f

# constrained triple gaussian - assumes third peak comes from sampling second peak twice
# can be used to fit noise, single- and double-PE peaks in SPE calibration
def gaussian_3(x, *p):
    
    A0, A1, A2, mu_0, mu_1, sigma_0, sigma_1 = p
    
    p0 = [A0, mu_0, sigma_0]
    p1 = [A1, mu_1, sigma_1]
    p2 = [A2, 2*mu_1, np.sqrt(2)*sigma_1] # standard deviations add in quadrature
    
    f = gaussian(x, *p0) + gaussian(x, *p1) + gaussian(x, *p2)
    
    return f

# constrained quadruple gaussian - assumes higher peaks come from sampling second peak multiple times
# can be used to fit noise and 1/2/3 PE peaks in SPE calibration
def gaussian_4(x, *p):
    
    A0, A1, A2, A3, mu_0, mu_1, sigma_0, sigma_1 = p
    
    p0 = [A0, mu_0, sigma_0]
    p1 = [A1, mu_1, sigma_1]
    p2 = [A2, 2*mu_1, np.sqrt(2)*sigma_1] # standard deviations add in quadrature
    p3 = [A3, 3*mu_1, np.sqrt(3)*sigma_1]
    
    f = gaussian(x, *p0) + gaussian(x, *p1) + gaussian(x, *p2) + gaussian(x, *p3)
    
    return f

# poisson distribution - free amplitude and expected value
# can be used to check accuracy of multi-PE fits in SPE calibration
def poisson(x, *p):
    
    x = np.array(x)
    A, l = p
    
    return A*l**x/scipy.special.factorial(x)*np.exp(-l)

def linearPlusGaussian(x, *p):
    
    A, mu, sigma, m, b = p
    
    return A*np.exp(-(x-mu)**2/(2.*sigma**2)) + m*x + b

######################
## Unit conversions ##
######################

# ADC counts -> mV
def adcTomV(array, dVperAdc):
    
    # array - nD array of raw ADC values from digitzer
    # dVperAdc - voltage step per ADC count = 2.25/2**14 (voltage range/number of ADC values in 14-bit digitization)
    
    array = np.array(array)
    
    return array*(dVperAdc*1000)

# SPE area in (ADC Counts)*samples -> gain
def areaToGain(array, dVperAdc, resistance, samplingFreq):
    
    # array - nD array of pulse areas in units of (ADC Counts)*samples
    # dVperAdc - voltage step per ADC count
    # resistance - PMT HV/signal line resistance
    # samplingFreq - digitizer sampling frequency (100 MHz -> 1e8)
    
    array = np.array(array)
    fundamentalCharge = 1.602e-19 # coulomb charge per electron
    
    return array*(dVperAdc/resistance/samplingFreq/fundamentalCharge)

########################
## Analysis functions ##
########################

# compute the gain of each PMT using SPE sizes
def calcGains(voltages, gainFile):
    
    # gainFile - .csv file with gain fit parameters, generated from relative gain calibration
    
    voltages = np.array(voltages)
    params = np.genfromtxt(gainFile, delimiter = ',', skip_header = 1) # read fit params file
    gains = 10**(params[:, 1] + params[:, 2])*voltages**params[:, 0] # Gain = 10^(log(A) + offset)*V^p
    
    return gains

# calculate scale factors to convert ADC counts to PE/sample for each channel
def scaleToPE(dVperAdc, resistance, samplingFreq, gains):
    
    # dVperAdc - voltage step per ADC count
    # resistance - PMT HV/signal line resistance
    # samplingFreq - digitizer sampling frequency (100 MHz -> 1e8)
    # gains - gain for each PMT, computed by calcGains
    
    fundamentalCharge = 1.602e-19 # coulomb charge per electron
    PE = dVperAdc/(resistance*samplingFreq*fundamentalCharge*gains)
    
    return PE

# perform baseline subtraction for each event in a channel
def adjBaseline(array, numSamples):
    
    # array - 2D array, 10,000 events with microseconds*100 samples/event
    # numSamples - number of samples used to compute baseline
    
    adj = np.array(array - np.average(array[:, 0:numSamples], axis = 1)) # subtract baseline avg. for each event
    
    return adj

# compute 3-sample average for a channel
def threeSampleAvg(array):
    
    # array - 2D array, 10,000 events with microseconds*100 samples/event
    
    stacked = np.array([np.delete(array, [0, -1], 1), np.delete(array, [-2, -1], 1), np.delete(array, [0, 1], 1)]) # stack arrays shifted -1, 0, 1 samples forward
    avg = np.average(stacked, axis = 0) # calculate average for each sample
    
    return avg

# find boundaries between pulses for a given channel and event
def findPulses(pulseIndices, event, absoluteSpacing, relativeSpacing):
    
    # pulseIndices - parallel arrays, [0] contains event indices, [1] contains sample indices meeting PE/sample threshold
    # event - event in file to look at
    # absoluteSpacing - pulses must be separated by this many samples
    # relative Spacing - pulses must be seperated by this fraction of the summed length of all previous pulses
    
    cutIndices = pulseIndices[1][np.where(pulseIndices[0] == event)] # select sample indices corresponding to the desired event
    if len(cutIndices) < 2: # we are only interested in pulses >= 20 ns (2 samples)
        return None, None
    
    indexDiff = cutIndices[1:] - cutIndices[:-1] # calculate spacing between indices meeting PE/sample threshold
    absoluteBoundaries = np.where(indexDiff >= absoluteSpacing)[0] + 1 # define boundaries where spacing > absolute criterion
    
    if len(absoluteBoundaries) == 0: # if there is only one pulse in the event...
        pulseIntervals = [0, 3000] # look at all cutIndices for that event
    else:
        relativeSpaceCut = (indexDiff[indexDiff >= absoluteSpacing]/absoluteBoundaries > relativeSpacing).flatten() # check if boundaries also meet relative criterion
        pulseBreaks = absoluteBoundaries[relativeSpaceCut] # select boundaries that meet both criteria
        pulseIntervals = np.insert(pulseBreaks, [0, len(pulseBreaks)], [0, 3000]) # insert starting/ending bounds for first/last pulses
        
    return cutIndices, pulseIntervals

def constructRQArray(options):
    
    numRQs = np.array(options)*np.array([7, 3, 9, 3, 3, 3]) + 3
    outData = outData = [[] for i in range(len(numRQs))]
    
    return outData

def constructRQList(options):
    
    RQList = []
    
    if options[0] == True:
        RQList.extend(['ch0Area', 'ch1Area', 'ch2Area', 'ch3Area', 'topSumArea', 'botArea', 'totalArea'])
    
    if options[1] == True:
        RQList.extend(['topSumHeight', 'botHeight', 'totalHeight'])
        
    if options[2] == True:
        RQList.extend(['topSumWidth', 'botWidth', 'totalWidth'])
        RQList.extend(['topSum05', 'bot05', 'total05'])
        RQList.extend(['topSum95', 'bot95', 'total95'])
        
    if options[3] == True:
        RQList.extend(['topSumRMS', 'botRMS', 'totalRMS'])
        
    if options[4] == True:
        RQList.extend(['topSumFw1050', 'botFw1050', 'totalFw1050'])
        
    if options[5] == True:
        RQList.extend(['topSumPf200', 'botPf200', 'totalPf200'])
        
    return RQList

def calcRQs(outData, ch0Wf, ch1Wf, ch2Wf, ch3Wf, topSumWf, botWf, totalWf, options):
    
    return

def applyDataCut(dictionary, mask):
    
    newDict = dict(dictionary)
    for key in newDict.keys():
        
        newDict[key] = newDict[key][mask]
    
    return newDict

def selectNthPulses(dictionary, pulse):
    
    values, indices = np.unique(dictionary['eventNum'], return_index = True)
    indices = indices + (pulse - 1)
    nthPulses = applyDataCut(dictionary, indices)
    
    return nthPulses

def reconstructPosition(dictionary, cartCoor):
    
    XPulsesPos = cartCoor*(dictionary['ch1Area'] - dictionary['ch0Area'] - dictionary['ch3Area'] + \
                           dictionary['ch2Area'])/dictionary['topSumArea']
    YPulsesPos = cartCoor*(dictionary['ch1Area'] + dictionary['ch0Area'] - dictionary['ch3Area'] - \
                           dictionary['ch2Area'])/dictionary['topSumArea']
    
    return np.array(XPulsesPos), np.array(YPulsesPos)

def calcPositionMeanVals(dictionary, key, XPos, YPos, xBins, yBins, cartCoor):
    
    keyList = [[[] for i in range(len(xBins))] for j in range(len(yBins))]
    
    for event in range(len(dictionary[key])):
        
        xIndex = np.searchsorted(xBins, XPos[event], side = 'right') - 1
        yIndex = np.searchsorted(yBins, YPos[event], side = 'right') - 1
        
        keyList[yIndex][xIndex].append(dictionary[key][event])
    
    keyMeans = ak.mean(ak.Array(keyList), axis = -1)
    keyFilled = ak.fill_none(keyMeans, 0)
    
    return keyFilled

def drawEllipseSelection(ratio, theta):
    
    t = np.linspace(0, 2*np.pi, 100)
    
    xEllipse = ratio*np.cos(t)*np.cos(theta) + np.sin(t)*np.sin(theta)
    yEllipse = -ratio*np.cos(t)*np.sin(theta) + np.sin(t)*np.cos(theta)
    
    return xEllipse, yEllipse

def cutEllipseSelection(S1Dict, S2Dict, ratio, theta, xCenter, yCenter, xScale, yScale):
    
    ellipseMask = ((S1Dict['botArea'] - xCenter)/xScale*np.cos(-theta) + \
                   (S2Dict['topSumArea'] - yCenter)/yScale*np.sin(-theta))**2/ratio**2 + \
                  (-(S1Dict['botArea'] - xCenter)/xScale*np.sin(-theta) + \
                   (S2Dict['topSumArea'] - yCenter)/yScale*np.cos(-theta))**2 <= 1
    
    S1Ellipse = applyDataCut(S1Dict, ellipseMask)
    S2Ellipse = applyDataCut(S2Dict, ellipseMask)
    
    return S1Ellipse, S2Ellipse

def calcAnticorrelationParams(S1Dict, S2Dict):
    
    dataSet = np.transpose(np.vstack((S1Dict['botArea'], S2Dict['topSumArea'])))
    clf = mixture.GaussianMixture(n_components = 1, covariance_type = "full")
    clf.fit(dataSet)
    
    v, w = scipy.linalg.eigh(clf.covariances_[0])
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    u = w[0]/scipy.linalg.norm(w[0])
    
    theta = np.arctan(u[1]/u[0])
    xMean = clf.means_[0][0]
    yMean = clf.means_[0][1]
    
    return xMean, yMean, theta

def calcMicroG1G2(xIntercept, yIntercept, energy, W):
    
    quanta = energy/W
    
    g1 = xIntercept/quanta
    g2 = yIntercept/quanta
    
    return g1, g2

def calcEventEnergies(S1Dict, S2Dict, g1, g2, W):
    
    E = W*(S1Dict['botArea']/g1 + S2Dict['topSumArea']/g2)
    
    return E

def calcElectronLifetime(driftTimes, S2Dict, driftMin, driftMax, fitMin, fitMax):
    
    tS2Sizes = np.log(S2Dict['topSumArea'])
    rangeCut = (driftTimes >= driftMin) & (driftTimes <= driftMax)
    seltS2Sizes = tS2Sizes[rangeCut]
    
    driftFit = np.polyfit(driftTimes, seltS2Sizes, 1)
    xRange = np.arange(fitMin, fitMax, 0.1)
    yRange = driftFit[0]*xRange + driftFit[1]
    
    return driftFit, xRange, yRange

def prepChannelData(dataFile, gainFile, voltages, dVperAdc, resistance, samplingFreq, baselineSamples, PEThreshold):
    
    gains = calcGains(voltages, gainFile)
    scaleFactorsPE = scaleToPE(dVperAdc, resistance, samplingFreq, gains)
    
    rootFile = uproot.open(dataFile)
    
    ch0 = rootFile['Channel0/ADCValues'].array()
    ch0Adj = adjBaseline(ch0, baselineSamples)*scaleFactorsPE[0]
    ch1 = rootFile['Channel1/ADCValues'].array()
    ch1Adj = adjBaseline(ch1, baselineSamples)*scaleFactorsPE[1]
    ch2 = rootFile['Channel2/ADCValues'].array()
    ch2Adj = adjBaseline(ch2, baselineSamples)*scaleFactorsPE[2]
    ch3 = rootFile['Channel3/ADCValues'].array()
    ch3Adj = adjBaseline(ch3, baselineSamples)*scaleFactorsPE[3]
    ch4 = rootFile['Channel4/ADCValues'].array()
    ch4Adj = adjBaseline(ch4, baselineSamples)*scaleFactorsPE[4]
    
    botAdj = ch4Adj[:, 4:]
    topSumAdj = (ch0Adj + ch1Adj + ch2Adj + ch3Adj)[:, :-4]
    totalAdj = botAdj + topSumAdj

    ch0Avg = threeSampleAvg(ch0Adj[:, :-4])
    ch1Avg = threeSampleAvg(ch1Adj[:, :-4])
    ch2Avg = threeSampleAvg(ch2Adj[:, :-4])
    ch3Avg = threeSampleAvg(ch3Adj[:, :-4])
    botAvg = threeSampleAvg(botAdj)
    topSumAvg = threeSampleAvg(topSumAdj)
    totalAvg = threeSampleAvg(totalAdj)

    thresholdMask = totalAvg < -1*PEThreshold
    pulseIndices = np.nonzero(thresholdMask)
    
    return ch0Avg, ch1Avg, ch2Avg, ch3Avg, botAvg, topSumAvg, totalAvg, pulseIndices

##################
## File loading ##
##################

#return sorted file names of a specified type from a directory
def getFileList(filePath, fileType):
    
    # filePath - directory containing files to be sorted
    # fileType - file extension to look for, eg. '*.root'; for all files use '*'
    
    fileNames = sorted(filter(os.path.isfile, glob.glob(filePath + fileType)))
    
    return np.array(fileNames)

def getFileSizes(fileNames):
    
    fileSizes = [os.path.getsize(f) for f in fileNames]
    
    return np.array(fileSizes)

def loadData(selFiles, RQs):
    
    RQDict = {}
    fileNum = 0
    for file in selFiles:
    
        for RQ in RQs:
        
            RQVals = uproot.open(file + ':' + RQ)['Values'].array()
            if RQ == 'eventNum':
                RQVals = RQVals + 10000*fileNum
        
            if fileNum == 0:
                RQDict[RQ] = RQVals
            else:
                RQDict[RQ] = ak.concatenate([RQDict[RQ], RQVals])
    
        fileNum += 1
        
    return RQDict

def loadRedFile(file, RQs):
    file = file
    RQDict = {}
    f = uproot.open(file) 
    for RQ in RQs:
        fileNum = 0
        RQVals = np.array(f['summary/' + RQ].array())
        if RQ == 'eventNum':
            RQVals = RQVals + 10000*fileNum
        if fileNum == 0:
            RQDict[RQ] = RQVals
        else:
            RQDict[RQ] = np.concatenate([RQDict[RQ], RQVals])
    return RQDict

def loadRawData(f,rawNames,scale):
    rawfile = uproot.open(rawNames[f])
    data0 = rawfile['Channel0/ADCValues'].array()
    data1 = rawfile['Channel1/ADCValues'].array()
    data2 = rawfile['Channel2/ADCValues'].array()
    data3 = rawfile['Channel3/ADCValues'].array()
    data4 = rawfile['Channel4/ADCValues'].array()
    
    data0_adj = data0 - np.average(data0[:, 0:100], axis = 1)
    data1_adj = data1 - np.average(data1[:, 0:100], axis = 1)
    data2_adj = data2 - np.average(data2[:, 0:100], axis = 1)
    data3_adj = data3- np.average(data3[:, 0:100], axis = 1)
    data4_adj = data4 - np.average(data4[:, 0:100], axis = 1)
    
    if type(scale) is str:
        d0 = data0_adj[:, :-4]
        d1 = data1_adj[:, :-4]
        d2 = data2_adj[:, :-4]
        d3 = data3_adj[:, :-4]
  
        topsum = d0 + d1 + d2 + d3
        bottom = data4_adj[:,4:]
        total = topsum+bottom
        x = np.arange(0, 2996, 1)
    elif np.isscalar(scale):
        d0 = data0_adj[:, :-4]*scale
        d1 = data1_adj[:, :-4]*scale
        d2 = data2_adj[:, :-4]*scale
        d3 = data3_adj[:, :-4]*scale
  
        topsum = d0 + d1 + d2 + d3
        bottom = data4_adj[:,4:]*scale
        total = topsum+bottom
        x = np.arange(0, 2996, 1)
    else:
        d0 = data0_adj[:, :-4]*scale[0]
        d1 = data1_adj[:, :-4]*scale[1]
        d2 = data2_adj[:, :-4]*scale[2]
        d3 = data3_adj[:, :-4]*scale[3]
  
        topsum = d0 + d1 + d2 + d3
        bottom = data4_adj[:,4:]*scale[4]
        total = topsum+bottom
        x = np.arange(0, 2996, 1)
 
    return x, np.array(d0), np.array(d1), np.array(d2), np.array(d3), np.array(topsum), np.array(bottom), np.array(total)

def ViewEvent(selFiles, rawFiles, RQs, fileNum, eventNum, channels = 'tot', scale='ADC', printRQs = False, xrange = [], yrange=[]):
    #load raw waveform and RQs
    x, d0, d1, d2, d3, topsum, bottom, total = loadRawData(fileNum,rawFiles,scale)
    RQDict = loadRedFile(selFiles[fileNum], RQs)
    #plot waveform
    if len(xrange)>0:
        plt.xlim(xrange[0],xrange[1])
    if len(yrange)>0:
        plt.ylim(yrange[0],yrange[1])
    if type(scale) is str:
        plt.xlabel('samples')
        plt.ylabel('ADC counts')
    elif np.isscalar(scale):
        plt.xlabel('samples')
        plt.ylabel('mV')
    else:
        plt.xlabel('samples')
        plt.ylabel('PE')
    if channels == 'all':
        plt.plot(x,d0[eventNum],label='tPMT0')
        plt.plot(x,d1[eventNum]-10,label='tPMT1')
        plt.plot(x,d2[eventNum]-20,label='tPMT2')
        plt.plot(x,d3[eventNum]-30,label='tPMT3')
        plt.plot(x,bottom[eventNum]-50,label='bPMT')
        plt.legend(fontsize=12)
        plt.show()
    elif channels == 'tb':
        plt.plot(x,topsum[eventNum],label='topSum')
        plt.plot(x,bottom[eventNum]-20,label='bPMT')
        plt.legend(fontsize=12)
        plt.show()
    else:
        plt.plot(x,total[eventNum])
        plt.show()
    #print RQs
    if printRQs == True:
        eventInds = np.where(np.in1d(RQDict['eventNum'],eventNum))[0] 
        print('Pulses in event:',len(eventInds))
        if len(eventInds)==2:
            driftTime = RQDict['startTime'][eventInds[1]] - RQDict['endTime'][eventInds[0]]
            print('driftTime:',driftTime)
            print()
        for i in range(len(eventInds)):
            print('Pulse '+str(i+1)+':')
            for RQ in RQs:
                print(str(RQ)+':',RQDict[str(RQ)][eventInds[i]])
            print()
            
##########################################
## PMT calibration and characterization ##
##########################################

def LED_simple(fileNames, start, stop, channel):
    
    areas = []
    heights = []
    offsets = []
    for n in range(len(fileNames)):
        root_file = uproot.open(fileNames[n])
        
        PMT = root_file['Channel' + str(channel) + '/ADCValues'].array()
        PMT_adj = PMT - np.average(PMT[:, 0:100], axis = 1)
        
        for i in range(len(PMT_adj)):
            total_wf = np.array(PMT_adj[i][start:stop])
            area = -1*sum(total_wf)
            height = -1*np.amin(total_wf)
            offset = np.amax(total_wf)    
            areas.append(area)
            heights.append(height)
            offsets.append(offset)
    
    return areas, heights, offsets

def compute_area(*p):
    A, mu, sigma = p
    x = np.arange(-5*sigma, 5*sigma, 1) + mu
    fit = gaussian(x, *p)
    return sum(fit)

def PMTPulseArea(fileNames, channel, start, cutoff, pulseArea):
    for fileName in fileNames:
        file = uproot.open(fileName)
        adc = file['Channel'+str(channel)+'/ADCValues'].array()
        ##Take the average of the first 100 adc values for each event, and shift everything by that
        ##This will make it easier to integrate later, since we can just sum the adjusted adc values
        adc_adj = adc - np.average(adc[:, 0:100], axis = 1)
        
        for event in range(len(adc_adj)):
            ##Find the first adc value before 1600 counts that is positive, indicating the end of the signal
            try:
                interval = np.where(adc_adj[event][cutoff:1600] >= 0)[0][0]
            except:
                print(fileName, event)
            end = cutoff + interval
            ##Integrate over the pulse area. Multiplying by -1 since the pulse is negative
            pulseArea.append(-1*sum(adc_adj[event][start:end]))
    return pulseArea

def count_pulses(fileNames, threshold, channel):
    
    total_pulses = []
    for fileName in fileNames:
        
        root_file = uproot.open(fileName)
        PMT = root_file['Channel' + str(channel) + '/ADCValues'].array()
        PMT_adj = PMT - np.average(PMT[:, 0:100], axis = 1)
        
        for event in PMT_adj:
            
            trigger = np.nonzero(event[0:1400] < threshold/(2.25/2**14*1000))[0]
            spaces = trigger[1:] - trigger[:-1]
            pulses = sum(spaces > 1)
            if len(trigger) > 0:
                pulses += 1
            total_pulses.append(pulses)
    
    return total_pulses

#################################
## Trigger rate and levelmeter ##
#################################

# remove unwanted trigger data before plotting
def prepTriggerData(triggerFile):
    
    # triggerFile - .txt file with trigger rates for each PMT + corresponding unix times
    # Note - trigger rates in the first and last second of each file need to be discarded because they do not represent
    #        a full second of data taking
    
    triggerData = np.genfromtxt(triggerFile, delimiter = ' ') # read trigger data
    goodVals = np.delete(triggerData, [0, len(triggerData) - 1], axis = 0) # cut first and last second from the data
    times = goodVals[:, 0] - np.amin(goodVals[:, 0]) # compute times since data-taking start
    
    for i in range(len(times) - 1, 0, -1): # start at the end of the file and work backwards
        if times[i] - times[i - 1] > 1: # if time spacing between data points > 1 second (aka new file)...
            goodVals = np.delete(goodVals, [i - 1, i], axis = 0) # delete data points on both sides
            
    adjTimes  = goodVals[:, 0] - np.amin(goodVals[:, 0]) # recompute times for selected data only
            
    return goodVals, adjTimes

# get the detector liquid level at the given unix times
def getLiquidLevels(timeStamps, levelFile, startUnixTime):
    
    # timeStamps - unix times at which liquid levels are calculated
    # levelData - 2D array of time in seconds *SINCE SLOW CONTROL DATABASE INITIALIZED* and level values at those times
    # startUnixTime - unix time of first value in levelData, computed manually
    
    levelData = np.genfromtxt(levelFile, delimiter = ',', skip_header = 1)[:, 1:] # read level data
    
    levelTimes = levelData[:, 0] # get database times
    levelVals = levelData[:, 1] # get level values
    adjLevelTimes = (levelTimes - levelTimes[0]) + startUnixTime # convert database times to unix times
    
    finalLevels = [] # initialize return array for liquid levels
    for timeStamp in timeStamps: # for each time stamp...
        
        index = np.searchsorted(adjLevelTimes, timeStamp, side = 'right') # adjLT[index - 1] <= timeStamp < adjLT[index]
    
        if timeStamp == adjLevelTimes[index - 1]: # check edge condition caused due to <= sign in np.searchsorted
            finalLevels.append(levelVals[index - 1])
        else:
            levelStep = levelVals[index] - levelVals[index - 1] # compute step between closest database values
            frac = (timeStamp - adjLevelTimes[index - 1])/(adjLevelTimes[index] - adjLevelTimes[index - 1]) # percent of the way to next database value
            compLevel = levelVals[index - 1] + levelStep*frac # linear interpolation between values
            finalLevels.append(compLevel)
            
    return np.array(finalLevels)

def getFileUnixTimes(fileNames, relative  = False):
    
    unixTimes = []
    for file in fileNames:
    
        dateIndex = file.find('MiX_')
        date = file[dateIndex + 4:dateIndex + 12]
    
        timeIndex = file.find('T', dateIndex)
        time = file[timeIndex + 1: timeIndex + 7]
    
        dateTime = datetime.datetime(int(date[0:4]), int(date[4:6]), int(date[6:8]), 
                                 int(time[0:2]), int(time[2:4]), int(time[4:6]))
        unixTime = timelib.mktime(dateTime.timetuple())
        unixTimes.append(unixTime)
        
    unixTimes = np.array(unixTimes)
    if relative:
        
        unixTimes = unixTimes - unixTimes[0]
    
    return unixTimes
    
