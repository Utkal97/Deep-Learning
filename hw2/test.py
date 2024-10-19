import sys
import os
import json
import pickle
import torch
import numpy as np
import math
from functools import reduce
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from models import Attention, Encoder, Decoder, Models

modelFilePath = 'model.h5'
model = torch.load(modelFilePath, map_location=lambda storage, loc: storage, weights_only=False)
print(model)

class VideoDataset(Dataset):
    def __init__(self, datasetPath):
        self.videoData = []
        for fileName in os.listdir(datasetPath):
            if fileName.endswith('.npy'):
                videoId = fileName.split('.npy')[0]
                videoFeatures = np.load(os.path.join(datasetPath, fileName))
                self.videoData.append((videoId, videoFeatures))
        
    def __len__(self):
        return len(self.videoData)

    def __getitem__(self, index):
        return self.videoData[index]

testDatasetPath = sys.argv[1]
videoDataset = VideoDataset(testDatasetPath)
videoLoader = DataLoader(videoDataset, batch_size=64, shuffle=True)

wordToIndexPath = 'wordOfIndex.pickle'
with open(wordToIndexPath, 'rb') as handle:
    wordToIndex = pickle.load(handle)

def generatePredictions(dataLoader, model, wordToIndex):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batchIndex, (videoIds, videoFeatures) in enumerate(dataLoader):
            videoFeatures = videoFeatures.float().cuda()
            _, predictedSequences = model(videoFeatures, mode='inference')
            
            for videoId, predictedSequence in zip(videoIds, predictedSequences):
                sentence = ' '.join(wordToIndex.get(wordId.item(), 'unknown') for wordId in predictedSequence)
                sentence = sentence.split('<EOS>')[0]
                predictions.append((videoId, sentence))

    return predictions

model = model.cuda()
predictions = generatePredictions(videoLoader, model, wordToIndex)

outputFilePath = sys.argv[2]
with open(outputFilePath, 'w') as outputFile:
    for videoId, caption in predictions:
        outputFile.write(f'{videoId},{caption}\n')

with open("testing_label.json", 'r') as file:
    testLabels = json.load(file)

predictedCaptions = {}
with open(outputFilePath, 'r') as outputFile:
    for line in outputFile:
        videoId, caption = line.rstrip().split(',', 1)
        predictedCaptions[videoId] = caption
        
def calculateBrevityPenalty(candidateLength, referenceLength):    
    return 1 if candidateLength > referenceLength else math.exp(1 - (float(referenceLength) / candidateLength))

def calculateNgram(candidateSentences, referenceSentences, n):
    clippedCount, totalCount, totalReferenceLength, totalCandidateLength = 0, 0, 0, 0
    
    for sentenceIndex, candidateSentence in enumerate(candidateSentences):
        referenceCounts = []
        referenceLengths = []

        for reference in referenceSentences:
            referenceSentence = reference[sentenceIndex].strip().split()
            referenceLengths.append(len(referenceSentence))
            ngramCounts = defaultdict(int)
            
            for i in range(len(referenceSentence) - n + 1):
                ngram = ' '.join(referenceSentence[i:i+n]).lower()
                ngramCounts[ngram] += 1
            referenceCounts.append(ngramCounts)

        candidateWords = candidateSentence.strip().split()
        candidateCounts = defaultdict(int)
        for i in range(len(candidateWords) - n + 1):
            ngram = ' '.join(candidateWords[i:i+n]).lower()
            candidateCounts[ngram] += 1

        count = sum(min(candidateCounts[ngram], max(refCounts.get(ngram, 0) for refCounts in referenceCounts))
                      for ngram in candidateCounts)

        clippedCount += count
        totalCount += max(1, len(candidateWords) - n + 1)
        
        bestMatchLength = min(referenceLengths, key=lambda refLength: abs(refLength - len(candidateWords)))
        totalReferenceLength += bestMatchLength
        totalCandidateLength += len(candidateWords)

    precision = float(clippedCount) / totalCount if totalCount > 0 else 0
    penalty = calculateBrevityPenalty(totalCandidateLength, totalReferenceLength)
    
    return precision, penalty

def calculateBleuScore(candidateSentence, referenceSentences, isMultipleReferences=False):
    candidateSentence = [candidateSentence.strip()]
    referenceSentences = [[ref.strip()] for ref in referenceSentences] if isMultipleReferences else [[referenceSentences.strip()]]
    
    precision, penalty = calculateNgram(candidateSentence, referenceSentences, 1)
    score = precision * penalty
    return score

bleuScores = []
for item in testLabels:
    captions = [caption.rstrip('.') for caption in item['caption']]
    bleuScore = calculateBleuScore(predictedCaptions[item['id']], captions, isMultipleReferences=True)
    bleuScores.append(bleuScore)

averageBleuScore = sum(bleuScores) / len(bleuScores)
print(f"Average BLEU score is {averageBleuScore}")
