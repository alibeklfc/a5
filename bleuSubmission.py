import csv
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

def tokenizeFile(fileName, flgList = 0):
	with open(fileName, newline='') as f:

		count =0

		spamreader = f.readlines()
		print(spamreader[0:3])

		listOfSent = []
		for line in spamreader[0:3]:
			arrTokens = line.replace('\n', '').split(",")
			print("Tokens:" , line,  arrTokens)

			if flgList:
				listOfSent.append([arrTokens])
			else:
				listOfSent.append(arrTokens)

	return listOfSent

listOfRef = tokenizeFile('tokensTrain.txt', flgList=1)

# This is your submissions' predictions

listOfHypo = tokenizeFile('tokensTestPredicted.txt')

print(len(listOfRef), len(listOfHypo))

print((listOfRef[0], listOfHypo[0]))

bleu_dic = {}
bleu_dic['1-grams'] = corpus_bleu(listOfRef, listOfHypo, weights=(1.0, 0, 0, 0))
bleu_dic['1-2-grams'] = corpus_bleu(listOfRef, listOfHypo, weights=(0.5, 0.5, 0, 0))

print(bleu_dic)