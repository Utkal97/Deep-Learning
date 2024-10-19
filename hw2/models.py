import torch.optim as optim
import time
import torch
from torch.autograd import Variable


class Attention(torch.nn.Module):
    def __init__(self, dimension):
        super(Attention, self).__init__()
        self.dimension = dimension

        self.firstLayer = torch.nn.Linear(2 * dimension, dimension)
        self.secondLayer = torch.nn.Linear(dimension, dimension)
        self.thirdLayer = torch.nn.Linear(dimension, dimension)
        self.fourthLayer = torch.nn.Linear(dimension, dimension)
        self.fifthLayer = torch.nn.Linear(dimension, dimension)
    
        self.linearLayer = torch.nn.Linear(dimension, 1, bias=False)

    def forward(self, hiddenState, encoderOutputs):
        batchSize, seqLen, featureCount = encoderOutputs.size()
        hiddenState = hiddenState.view(batchSize, 1, featureCount).repeat(1, seqLen, 1)
        matchingInputs = torch.cat((encoderOutputs, hiddenState), 2).view(-1, 2 * self.dimension)

        x = self.firstLayer(matchingInputs)
        x = self.secondLayer(x)
        x = self.thirdLayer(x)
        x = self.fourthLayer(x)
        x = self.fifthLayer(x)

        attentionWeights = self.linearLayer(x)
        attentionWeights = attentionWeights.view(batchSize, seqLen)
        attentionWeights = torch.nn.functional.softmax(attentionWeights, dim=1)
        context = torch.bmm(attentionWeights.unsqueeze(1), encoderOutputs).squeeze(1)

        return context

class Encoder(torch.nn.Module):
    def __init__(self, inputSize=4096, hiddenSize=256, numLayers=1, dropout=0.30):
        super(Encoder, self).__init__()
        self.embeddingLayer = torch.nn.Linear(inputSize, hiddenSize)
        self.dropoutLayer = torch.nn.Dropout(dropout)
        self.lstmLayer = torch.nn.LSTM(hiddenSize, hiddenSize, num_layers=numLayers, batch_first=True)

    def forward(self, input):
        batchSize, seqLen, featureCount = input.size()
        input = input.view(-1, featureCount)
        embeddedInput = self.embeddingLayer(input)
        embeddedInput = self.dropoutLayer(embeddedInput)
        embeddedInput = embeddedInput.view(batchSize, seqLen, -1)
        output, (hiddenState, cellState) = self.lstmLayer(embeddedInput)
        return output, hiddenState

class Decoder(torch.nn.Module):
    def __init__(self, hiddenSize, outputSize, vocabSize, wordDim, dropoutPercentage=0.30):
        super(Decoder, self).__init__()

        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.vocabSize = vocabSize
        self.wordDim = wordDim

        self.embeddingLayer = torch.nn.Embedding(outputSize, wordDim)
        self.dropoutLayer = torch.nn.Dropout(dropoutPercentage)
        self.lstmLayer = torch.nn.LSTM(hiddenSize + wordDim, hiddenSize, batch_first=True)
        self.attentionLayer = Attention(hiddenSize)
        self.finalOutputLayer = torch.nn.Linear(hiddenSize, vocabSize)

    def forward(self, encoderLastHiddenState, encoderOutput, targets=None, mode='train', trainingSteps=None):
        batchSize = encoderLastHiddenState.size(1)

        decoderCurrentHiddenState = None if encoderLastHiddenState is None else encoderLastHiddenState
        decoderContext = torch.zeros(decoderCurrentHiddenState.size()).cuda()

        decoderCurrentInputWord = Variable(torch.ones(batchSize, 1)).long().cuda()
        seqLogProb = []
        seqPredictions = []

        targets = self.embeddingLayer(targets)
        seqLen = targets.size(1)

        for i in range(seqLen - 1):
            threshold = scipy.special.expit(trainingSteps / 20 + 0.85)
            if random.uniform(0.05, 0.995) > threshold:
                currentInputWord = targets[:, i]
            else:
                currentInputWord = self.embeddingLayer(decoderCurrentInputWord).squeeze(1)

            context = self.attentionLayer(decoderCurrentHiddenState, encoderOutput)
            lstmInput = torch.cat([currentInputWord, context], dim=1).unsqueeze(1)
            lstmOutput, t = self.lstmLayer(lstmInput, (decoderCurrentHiddenState, decoderContext))
            decoderCurrentHiddenState = t[0]
            logProb = self.finalOutputLayer(lstmOutput.squeeze(1))
            seqLogProb.append(logProb.unsqueeze(1))
            decoderCurrentInputWord = logProb.unsqueeze(1).max(2)[1]

        seqLogProb = torch.cat(seqLogProb, dim=1)
        seqPredictions = seqLogProb.max(2)[1]
        return seqLogProb, seqPredictions

    def inference(self, encoderLastHiddenState, encoderOutput):
        batchSize = encoderLastHiddenState.size(1)
        decoderCurrentHiddenState = None if encoderLastHiddenState is None else encoderLastHiddenState
        decoderCurrentInputWord = Variable(torch.ones(batchSize, 1)).long().cuda()
        decoderContext = torch.zeros(decoderCurrentHiddenState.size()).cuda()
        seqLogProb = []
        seqPredictions = []
        assumptionSeqLen = 28

        for i in range(assumptionSeqLen - 1):
            currentInputWord = self.embeddingLayer(decoderCurrentInputWord).squeeze(1)
            context = self.attentionLayer(decoderCurrentHiddenState, encoderOutput)
            lstmInput = torch.cat([currentInputWord, context], dim=1).unsqueeze(1)
            lstmOutput, t = self.lstmLayer(lstmInput, (decoderCurrentHiddenState, decoderContext))
            decoderCurrentHiddenState = t[0]
            logProb = self.finalOutputLayer(lstmOutput.squeeze(1))
            seqLogProb.append(logProb.unsqueeze(1))
            decoderCurrentInputWord = logProb.unsqueeze(1).max(2)[1]

        seqLogProb = torch.cat(seqLogProb, dim=1)
        seqPredictions = seqLogProb.max(2)[1]
        return seqLogProb, seqPredictions
    
class Models(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(Models, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, aviFeat, mode, targetSentences=None, trainingSteps=None):
        encoderOutputs, encoderLastHiddenState = self.encoder(aviFeat)

        if mode == 'train':
            seqLogProb, seqPredictions = self.decoder(
                encoderLastHiddenState=encoderLastHiddenState,
                encoderOutput=encoderOutputs,
                targets=targetSentences,
                mode=mode,
                trainingSteps=trainingSteps
            )
        elif mode == 'inference':
            seqLogProb, seqPredictions = self.decoder.inference(
                encoderLastHiddenState=encoderLastHiddenState,
                encoderOutput=encoderOutputs
            )

        return seqLogProb, seqPredictions