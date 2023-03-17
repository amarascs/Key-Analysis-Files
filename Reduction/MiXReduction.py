import numpy as np
import uproot
import awkward as ak
import array as arr
import sys
import json
import ROOT


# compute the gain of each PMT using SPE sizes
def calcGains(voltages, gainFile):
    
    # gainFile - .csv file with gain fit parameters

    voltages = np.array(voltages)
    params = np.genfromtxt(gainFile, delimiter = ',', skip_header = 1) # read fit params file
    gains = 10**(params[:, 1] + params[:, 2])*voltages**params[:, 0] # G = 10^(log(A) + offset)*V^p
    
    return gains

# calculate scale factors to convert ADC counts to PE/sample for each channel
def scaleToPE(dVperAdc, resistance, samplingFreq, gains):

    # dVperAdc - voltage step per ADC count
    # resistance - PMT HV/signal line resistance
    # samplingFreq - digitizer sampling frequency (100 MHz -> 1e8)
    # gains - gain for each PMT, computed by calcGains

    fundamentalCharge = 1.602e-19 #coulomb charge per electron
    PE = dVperAdc/(resistance*samplingFreq*fundamentalCharge*gains)

    return PE

# perform baseline subtraction for each event in a channel
def adjBaseline(array, numSamples):

    # array - 2D array, 10,000 events with 3,000 samples/event
    # numSamples - number of samples used to compute baseline

    adj = np.array(array - np.mean(array[:, 0:numSamples], axis = 1)) # subtract baseline avg. for each event

    return adj

# compute 3-sample average for a channel
def threeSampleAvg(array):
    
    # array - 2D array, 10,000 events with 3,000 samples/event
    
    stacked = np.array([np.delete(array, [0, -1], 1), np.delete(array, [-2, -1], 1), np.delete(array, [0, 1], 1)]) # stack arrays shifted -1, 0, 1 samples forward
    avg = np.average(stacked, axis = 0) # calculate average for each sample
    
    return avg

# find boundaries between pulses for a given channel and event
def splitPulses(pulseIndices, event, absoluteSpacing, relativeSpacing):
    
    # pulseIndices - parallel arrays, [0] contains event indices, [1] contains sample indices
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

def main():

    dataFile = sys.argv[1] # raw .root file to analyze
    outdir = sys.argv[2] # output directory
    paramsFile = sys.argv[3] # .json file containing analysis parameters

    with open(paramsFile) as f: # open json file

        paramsData = json.load(f) # load parameters object from file

        gainFile = paramsData['gainFile'] # read gain file path from object
        RQs = paramsData['RQs'] # selected RQs to produce
    
        dVperAdc = paramsData['dVperAdc'] # voltage step per ADC count
        resistance = paramsData['resistance'] # PMT HV/signal line resistance
        samplingFreq = paramsData['samplingFreq'] # digitizer sampling frequency
        voltages = paramsData['voltages'] # PMT voltages

        PEThreshold = paramsData['PEThreshold'] # pulse threshold in PE/sample
        baselineSamples = paramsData['baselineSamples'] # number of samples used to compute baseline
        absoluteSpacing = paramsData['absoluteSpacing'] # pulses must be separated by this many samples
        relativeSpacing = paramsData['relativeSpacing'] # pulses must be seperated by this fraction of the summed length of all previous pulses
        cartCoor = paramsData['cartCoor'] # distance of tPMTs from detector center (mm)

    outData = [[] for i in range(len(RQs))] # list of lists for each RQ

    gains = calcGains(voltages, gainFile) # calculate gains for each PMT
    scaleFactorsPE = scaleToPE(dVperAdc, resistance, samplingFreq, gains) # calculate scale factor from ADC counts to PE/sample

    rootFile = uproot.open(dataFile) # open data file

    ch0 = rootFile['Channel0/ADCValues'].array() # read ADC values from channel 0 (tPMT0)
    ch0Adj = adjBaseline(ch0, baselineSamples)*scaleFactorsPE[0] # subtract baseline on a per-event basis
    ch1 = rootFile['Channel1/ADCValues'].array() # channel 1 (tPMT1)
    ch1Adj = adjBaseline(ch1, baselineSamples)*scaleFactorsPE[1]
    ch2 = rootFile['Channel2/ADCValues'].array() # channel 2 (tPMT2)
    ch2Adj = adjBaseline(ch2, baselineSamples)*scaleFactorsPE[2]
    ch3 = rootFile['Channel3/ADCValues'].array() # channel 3 (tPMT3)
    ch3Adj = adjBaseline(ch3, baselineSamples)*scaleFactorsPE[3]
    ch4 = rootFile['Channel4/ADCValues'].array() # channel 4 (bPMT)
    ch4Adj = adjBaseline(ch4, baselineSamples)*scaleFactorsPE[4]

    botAdj = ch4Adj[:, 4:] # shift bPMT signal 4 samples (40 ns) earlier to correct for longer response time than the tPMTs
    topSumAdj = (ch0Adj + ch1Adj + ch2Adj + ch3Adj)[:, :-4] # cut last four samples from tPMTs to align channels
    totalAdj = botAdj + topSumAdj # compute total summed waveform, used for pulse finding

    ch0Avg = threeSampleAvg(ch0Adj[:, :-4]) # compute three-sample average for tPMT0
    ch1Avg = threeSampleAvg(ch1Adj[:, :-4]) # tPMT1
    ch2Avg = threeSampleAvg(ch2Adj[:, :-4]) # tPMT2
    ch3Avg = threeSampleAvg(ch3Adj[:, :-4]) # tPMT3
    botAvg = threeSampleAvg(botAdj) # bPMT
    topSumAvg = threeSampleAvg(topSumAdj) # top sum
    totalAvg = threeSampleAvg(totalAdj) # total

    thresholdMask = totalAvg < -1*PEThreshold # boolean mask of all samples where the PE/sample threshold is met
    pulseIndices = np.nonzero(thresholdMask) # parallel arrays, [0] contains event indices, [1] contains sample indices

    for event in range(len(totalAvg)): # for each event in the file (10,000 total)

        cutIndices, pulseIntervals = splitPulses(pulseIndices, event, absoluteSpacing, relativeSpacing) # select trigger indices and find pulse boundaries for the event
        if pulseIntervals is not None: # if there is at least one pulse in the event

            for pulse in range(len(pulseIntervals) - 1): # for each pulse found

                triggerRange = cutIndices[pulseIntervals[pulse]:pulseIntervals[pulse + 1]] # select trigger indices corresponding to the pulse
                if (len(triggerRange) < 2) or (np.amin(triggerRange) < 100): # pulse must be >= 20 ns and not occur in the first 100 samples (baseline region)
                    continue

                startTime = triggerRange[0] - 9
                endTime = triggerRange[-1] + 10

                ch0Wf = ch0Avg[event][startTime:endTime] # include 10 samples on either side of tPMT0 waveform
                ch1Wf = ch1Avg[event][startTime:endTime] # tPMT1
                ch2Wf = ch2Avg[event][startTime:endTime] # tPMT2
                ch3Wf = ch3Avg[event][startTime:endTime] #tPMT3
                totalWf = totalAvg[event][startTime:endTime] # total
                topSumWf = topSumAvg[event][startTime:endTime] # top sum
                botWf = botAvg[event][startTime:endTime] # bPMT

                if np.sum((topSumWf < -1*PEThreshold) * (botWf < -1*PEThreshold)) >= 2:
                    
                    ch0Area = -1*np.sum(ch0Wf)
                    ch1Area = -1*np.sum(ch1Wf)
                    ch2Area = -1*np.sum(ch2Wf)
                    ch3Area = -1*np.sum(ch3Wf)
                    topSumArea = -1*np.sum(topSumWf)
                    botArea = -1*np.sum(botWf)
                    totalArea = -1*np.sum(totalWf)
                    
                    ch0Height = -1*np.amin(ch0Wf)
                    ch1Height = -1*np.amin(ch1Wf)
                    ch2Height = -1*np.amin(ch2Wf)
                    ch3Height = -1*np.amin(ch3Wf)
                    topSumHeight = -1*np.amin(topSumWf)
                    botHeight = -1*np.amin(botWf)
                    totalHeight = -1*np.amin(totalWf)
                    
                    topSumCS = -1*np.cumsum(topSumWf)
                    botCS = -1*np.cumsum(botWf)
                    totalCS = -1*np.cumsum(totalWf)
                    
                    topSum05 = (topSumCS/topSumArea >= 0.05).argmax() + startTime
                    topSum50 = (topSumCS/topSumArea >= 0.5).argmax() + startTime
                    topSum95 = (topSumCS/topSumArea >= 0.95).argmax() + startTime
                    bot05 = (botCS/botArea >= 0.05).argmax() + startTime
                    bot50 = (botCS/botArea >= 0.5).argmax() + startTime
                    bot95 = (botCS/botArea >= 0.95).argmax() + startTime
                    total05 = (totalCS/totalArea >= 0.05).argmax() + startTime
                    total50 = (totalCS/totalArea >= 0.5).argmax() + startTime
                    total95 = (totalCS/totalArea >= 0.95).argmax() + startTime
                    
                    topSumPulse = topSumWf[topSum05 - startTime:topSum95 - startTime]
                    botPulse = botWf[bot05 - startTime:bot95 - startTime]
                    totalPulse = totalWf[total05 - startTime:total95 - startTime]
                    topSumMean = np.sum(topSumPulse*np.arange(topSum05, topSum95, 1))/np.sum(topSumPulse)
                    botMean = np.sum(botPulse*np.arange(bot05, bot95, 1))/np.sum(botPulse)
                    totalMean = np.sum(totalPulse*np.arange(total05, total95, 1))/np.sum(totalPulse)
                    topSumRMS = np.sqrt(np.abs(np.sum((np.arange(topSum05, topSum95, 1) - topSumMean)**2*topSumPulse)/np.sum(topSumPulse)))
                    botRMS = np.sqrt(np.abs(np.sum((np.arange(bot05, bot95, 1) - botMean)**2*botPulse)/np.sum(botPulse)))
                    totalRMS = np.sqrt(np.abs(np.sum((np.arange(total05, total95, 1) - totalMean)**2*totalPulse)/np.sum(totalPulse)))
                    
                    topSumWidth = topSum95 - topSum05
                    botWidth = bot95 - bot05
                    totalWidth = total95 - total05
                    
                    outData[0].append(ch0Area)
                    outData[1].append(ch1Area)
                    outData[2].append(ch2Area)
                    outData[3].append(ch3Area)
                    outData[4].append(topSumArea)
                    outData[5].append(botArea)
                    outData[6].append(totalArea)
                    outData[7].append(ch0Height)
                    outData[8].append(ch1Height)
                    outData[9].append(ch2Height)
                    outData[10].append(ch3Height)
                    outData[11].append(topSumHeight)
                    outData[12].append(botHeight)
                    outData[13].append(totalHeight)
                    outData[14].append(topSumWidth)
                    outData[15].append(botWidth)
                    outData[16].append(totalWidth)
                    outData[17].append(topSumRMS)
                    outData[18].append(botRMS)
                    outData[19].append(totalRMS)
                    outData[20].append(topSum05)
                    outData[21].append(bot05)
                    outData[22].append(total05)
                    outData[23].append(topSum50)
                    outData[24].append(bot50)
                    outData[25].append(total50)
                    outData[26].append(topSum95)
                    outData[27].append(bot95)
                    outData[28].append(total95)
                    outData[29].append(startTime)
                    outData[30].append(endTime)
                    outData[31].append(event)

    outputname = outdir + '/' + str('red') + dataFile[-25:]
    outFile = ROOT.TFile( outputname, 'recreate' )
    t = ROOT.TTree( 'summary', 'summary' )

    ch0Area_      = arr.array('d', [0.])
    ch1Area_      = arr.array('d', [0.])
    ch2Area_      = arr.array('d', [0.])
    ch3Area_      = arr.array('d', [0.])
    topSumArea_   = arr.array('d', [0.])
    botArea_      = arr.array('d', [0.])
    totalArea_    = arr.array('d', [0.])
    ch0Height_    = arr.array('d', [0.])
    ch1Height_    = arr.array('d', [0.])
    ch2Height_    = arr.array('d', [0.])
    ch3Height_    = arr.array('d', [0.])
    topSumHeight_ = arr.array('d', [0.])
    botHeight_    = arr.array('d', [0.])
    totalHeight_  = arr.array('d', [0.])
    topSumWidth_  = arr.array('d', [0.])
    botWidth_     = arr.array('d', [0.])
    totalWidth_   = arr.array('d', [0.])
    topSumRMS_    = arr.array('d', [0.])
    botRMS_       = arr.array('d', [0.])
    totalRMS_     = arr.array('d', [0.])
    topSum05_     = arr.array('d', [0.])
    bot05_        = arr.array('d', [0.])
    total05_      = arr.array('d', [0.])
    topSum50_     = arr.array('d', [0.])
    bot50_        = arr.array('d', [0.])
    total50_      = arr.array('d', [0.])
    topSum95_     = arr.array('d', [0.])
    bot95_        = arr.array('d', [0.])
    total95_      = arr.array('d', [0.])
    startTime_    = arr.array('d', [0.])
    endTime_      = arr.array('d', [0.])
    eventNum_     = arr.array('d', [0.])

    t.Branch('ch0Area',           ch0Area_     , 'ch0Area/D')
    t.Branch('ch1Area',           ch1Area_     , 'ch1Area/D')
    t.Branch('ch2Area',           ch2Area_     , 'ch2Area/D')
    t.Branch('ch3Area',           ch3Area_     , 'ch3Area/D')
    t.Branch('topSumArea',     topSumArea_  , 'topSumArea/D')
    t.Branch('botArea',           botArea_     , 'botArea/D')
    t.Branch('totalArea',       totalArea_   , 'totalArea/D')
    t.Branch('ch0Height',       ch0Height_   , 'ch0Height/D')
    t.Branch('ch1Height',       ch1Height_   , 'ch1Height/D')
    t.Branch('ch2Height',       ch2Height_   , 'ch2Height/D')
    t.Branch('ch3Height',       ch3Height_   , 'ch3Height/D')
    t.Branch('topSumHeight', topSumHeight_, 'topSumHeight/D')
    t.Branch('botHeight',       botHeight_   , 'botHeight/D')
    t.Branch('totalHeight',   totalHeight_ , 'totalHeight/D')
    t.Branch('topSumWidth',   topSumWidth_ , 'topSumWidth/D')
    t.Branch('botWidth',         botWidth_    , 'botWidth/D')
    t.Branch('totalWidth',     totalWidth_  , 'totalWidth/D')
    t.Branch('topSumRMS',       topSumRMS_   , 'topSumRMS/D')
    t.Branch('botRMS',             botRMS_      , 'botRMS/D')
    t.Branch('totalRMS',         totalRMS_    , 'totalRMS/D')
    t.Branch('topSum05',         topSum05_    , 'topSum05/D')
    t.Branch('bot05',               bot05_       , 'bot05/D')
    t.Branch('total05',           total05_     , 'total05/D')
    t.Branch('topSum50',         topSum50_    , 'topSum50/D')
    t.Branch('bot50',               bot50_       , 'bot50/D')
    t.Branch('total50',           total50_     , 'total50/D')
    t.Branch('topSum95',         topSum95_    , 'topSum95/D')
    t.Branch('bot95',               bot95_       , 'bot95/D')
    t.Branch('total95',           total95_     , 'total95/D')
    t.Branch('startTime',       startTime_   , 'startTime/D')
    t.Branch('endTime',           endTime_     , 'endTime/D')
    t.Branch('eventNum',         eventNum_    , 'eventNum/D')
        
    for i in range(len(outData[-1])):
        ch0Area_[0] = outData[0][i]
        ch1Area_[0] = outData[1][i]
        ch2Area_[0] = outData[2][i]
        ch3Area_[0] = outData[3][i]
        topSumArea_[0] = outData[4][i]
        botArea_[0] = outData[5][i]
        totalArea_[0] = outData[6][i]
        ch0Height_[0] = outData[7][i]
        ch1Height_[0] = outData[8][i]
        ch2Height_[0] = outData[9][i]
        ch3Height_[0] = outData[10][i]
        topSumHeight_[0] = outData[11][i]
        botHeight_[0] = outData[12][i]
        totalHeight_[0] = outData[13][i]
        topSumWidth_[0] = outData[14][i]
        botWidth_[0] = outData[15][i]
        totalWidth_[0] = outData[16][i]
        topSumRMS_[0] = outData[17][i]
        botRMS_[0] = outData[18][i]
        totalRMS_[0] = outData[19][i]
        topSum05_[0] = outData[20][i]
        bot05_[0] = outData[21][i]
        total05_[0] = outData[22][i]
        topSum50_[0] = outData[23][i]
        bot50_[0] = outData[24][i]
        total50_[0] = outData[25][i]
        topSum95_[0] = outData[26][i]
        bot95_[0] = outData[27][i]
        total95_[0] = outData[28][i]
        startTime_[0] = outData[29][i]
        endTime_[0] = outData[30][i]
        eventNum_[0] = outData[31][i]

        t.Fill()

    t.Write()
    outFile.Close()

    print( "Done!" )

if __name__=="__main__":
    main()
