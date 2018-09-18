from CFA import CFA
from Data import Data
from Models import Models
from time import sleep

def main():
	cfaUnigram = CFA(Models, Data('Tweets.csv').UnigramMatrix)
	print('Begin graphing.')
	cfaUnigram.graphRankScores()

if __name__ == '__main__':
	main()
