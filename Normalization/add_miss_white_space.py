# Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
from math import log
  
class White_Space_Fixer:
            
    def __init__(self, sentence):
        self.sentence = sentence
        words = open("C:\\Users\\James\\Documents\\wordbank.txt").read().split()
        wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
        self.wordcost = wordcost
        maxword = max(len(x) for x in words)
        self.maxword = maxword
    
    def infer_sent_whitespace(self):
        sentence = self.sentence
        print(sentence)
        sentence = sentence.lower()
        l = sentence.split()
        j = []
        
        def infer_spaces(s):  
            
            def best_match(i):
                candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
                return min((c + self.wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)
        
            """Uses dynamic programming to infer the location of spaces in a string
            without spaces."""
            # Build the cost array.
            
            cost = [0]
            for i in range(1,len(s)+1):
                c,k = best_match(i)
                cost.append(c)
            # Backtrack to recover the minimal-cost string.
            out = []
            i = len(s)
            while i>0:
                c,k = best_match(i)
                assert c == cost[i]
                out.append(s[i-k:i])
                i -= k
            return " ".join(reversed(out))
    
        for i in range(0,len(l)):
            if l[i][0] == '@':
                j.append(l[i])
            elif l[i][-1] in ['.' , '?' , '!' , ',' , ';', ':']:
                j.append(l[i])
            else:
                j.append(infer_spaces(l[i]))
    
        return ' '.join(j)
