from enum import Enum
import re

class ScoringMethod(Enum):
    accuracy = 1
    informationGain = 2

class RegexRule(object):
    def __init__(self, regex, phrase, matched, distribution, classValue, numCorrect, score):
        self.regex = regex
        self.partialRegex = re.compile(regex.pattern.replace("(^|^.* )", "").replace("($| .*$)", ""))
        self.phrase = phrase
        self.matched = matched
        self.distribution = distribution
        self.numCorrect = numCorrect
        self.score = score
    
    def matches(self, s):
        return self.regex.match(s)

class RegexClassifier(object):
    def __init__(self, stringAttributeIndex):
        self.regexPrefix = "(^|^.* )"
        self.regexSuffix = "($| .*$)"
        self.gap = " (\\S+ )"
        self.matchedPattern = "MATCHED_PATTERN"
        self.regexTokens = [".", "^", "$", "(", ")", "[", "]", "{", "}", "?", "+", "|", "*"]
        self.jumpLength = 2
        self.scoreThreshold = 1
        self.scoringMethod = ScoringMethod.accuracy
        self.stringAttributeIndex = stringAttributeIndex
        self.classIndex = -1
    
    def train(self, data):
        self.classIndex = data.classIndex
        self.regexRules = []
        prefixes = set()
        suffixes = set()
        currentRegexRules = []
        newRegexRules = []
        jumpLength = max(0, self.jumpLength)
        improved = False
        
        for word in self.__findTopWords(data, 1):
            currentRegexRules.append(self.__createRegexRule(word, data))
            
            while len(currentRegexRules) > 0:
                newRegexRules.clear()

                for regexRule in currentRegexRules:
                    if regexRule == None or regexRule.phrase == None:
                        continue
                    
                    improved = False
                    candidates = []
                    
                    for i in range(0, jumpLength + 1):
                        self.__findPrefixesAndSuffixes(regexRule, regexRule.matched, prefixes, suffixes, i)
                        self.__expandRegex(regexRule.phrase, candidates, prefixes, regexRule.matched, True, i)
                        self.__expandRegex(regexRule.phrase, candidates, suffixes, regexRule.matched, False, i)
                    
                    for newRegexRule in candidates:
                        newScore = newRegexRule.score
                        score = regexRule.score
                        
                        if newScore > score or (newScore == score and newRegexRule.numCorrect > regexRule.numCorrect):
                            newRegexRules.append(newRegexRule)
                            
                            improved = True
                    
                    if not improved and regexRule.score >= self.scoreThreshold:
                        self.regexRules.append(regexRule)
                
                currentRegexRules = newRegexRules.copy()
        
        for regexRule in self.regexRules:
            regexRule.matched = None
        
        self.__createDefaultRegexRule(data)
    
    def classify(self, instance):
        for regexRule in self.regexRules:
            if regexRule.matches(instance.values[self.stringAttributeIndex]):
                return regexRule.distribution
        
        return {}
    
    def setScoreThreshold(self, scoreThreshold):
        self.scoreThreshold = scoreThreshold
    
    def __createDefaultRegexRule(self, data):
        distribution = {}
        maxClass = None
        maxValue = 0
        total = 0
        
        for instance in data.instances:
            matched = False
            
            for regexRule in self.regexRules:
                if regexRule.matches(instance.values[self.stringAttributeIndex]):
                    matched = True

            if not matched:
                if instance.values[self.classIndex] in distribution:
                    distribution[instance.values[self.classIndex]] = distribution[instance.values[self.classIndex]] + instance.weight
                else:
                    distribution[instance.values[self.classIndex]] = instance.weight
                
                total = total + instance.weight
                
                if distribution[instance.values[self.classIndex]] > maxValue:
                    maxValue = distribution[instance.values[self.classIndex]]
                    maxClass = instance.values[self.classIndex]
        
        for cValue, count in distribution.items():
            distribution[cValue] = count / total
              
        self.regexRules.append(RegexRule(re.compile(".*"), ".*", None, distribution, maxClass, maxValue, distribution[maxClass]))
    
    def __createRegexRule(self, phrase, data):
        distribution = {}
        total = 0
        numCorrect = 0
        classValue = None
        regex = re.compile(self.regexPrefix + self.__formatRegex(phrase) + self.regexSuffix)
        matched = []
        
        for instance in data.instances:
            if regex.match(instance.values[self.stringAttributeIndex]):
                total = total + instance.weight
                
                if instance.values[self.classIndex] in distribution:
                    distribution[instance.values[self.classIndex]] = distribution[instance.values[self.classIndex]] + instance.weight
                else:
                    distribution[instance.values[self.classIndex]] = instance.weight
                
                if distribution[instance.values[self.classIndex]] > numCorrect:
                    numCorrect = distribution[instance.values[self.classIndex]]
                    classValue = instance.values[self.classIndex]
                
                matched.append(instance)
        
        if total > 0:
            for cValue, count in distribution.items():
                distribution[cValue] = count / total
            
            return RegexRule(regex, phrase, matched, distribution, classValue, numCorrect, distribution[cValue])
        
        return None
    
    def __expandRegex(self, phrase, candidates, affixes, data, usePrefixes, gapSize):
        formattedGap = " "
        regexRule = None
        
        if gapSize > 0:
            formattedGap = self.gap + "{" + str(gapSize) + "}"
        
        for affix in affixes:
            if usePrefixes:
                regexRule = self.__createRegexRule(affix + formattedGap + phrase, data)
            else:
                regexRule = self.__createRegexRule(phrase + formattedGap + affix, data)
            
            if regexRule != None:
                candidates.append(regexRule)
        
        return candidates
    
    def __findPrefixesAndSuffixes(self, regexRule, data, prefixes, suffixes, gapSize):
        prefixes.clear()
        suffixes.clear()

        for instance in data:
            sentence = regexRule.partialRegex.sub(self.matchedPattern, instance.values[self.classIndex])
            
            while self.matchedPattern in sentence:
                partialSentence = sentence[0:sentence.index(self.matchedPattern)].strip().split(" ")
                partialSentenceLength = len(partialSentence)
                
                if gapSize < partialSentenceLength:
                    prefixes.add(partialSentence[partialSentenceLength - 1 - gapSize])
                
                sentence = sentence[sentence.index(self.matchedPattern) + len(self.matchedPattern)].strip()
                partialSentence = sentence.split(" ")

                if gapSize < len(partialSentence):
                    suffixes.add(partialSentence[gapSize])
    
    def __findTopWords(self, data, numWords):
        topWords = []
        words = {}
        wordAccuracy = {}
        
        for instance in data.instances:
            for word in instance.values[self.stringAttributeIndex].split(" "):
                if word in words:
                    if instance.values[self.classIndex] in words[word]:
                        words[word][instance.values[self.classIndex]] = words[word][instance.values[self.classIndex]] + instance.weight
                    else:
                        words[word][instance.values[self.classIndex]] = instance.weight
                else:
                    words[word] = {instance.values[self.classIndex] : instance.weight}
        
        for word, distribution in words.items():
            maxCount = 0
            total = 0
            
            for classValue, count in distribution.items():
                total = total + count
                
                if count > maxCount:
                    maxCount = count
            
            wordAccuracy[word] = {"accuracy" : maxCount / total, "count" : maxCount}
        
        for i in range(0, numWords):
            maxWord = ""
            maxAccuracy = 0
            maxCount = 0
            
            for word, stats in wordAccuracy.items():
                if stats["accuracy"] > maxAccuracy or (stats["accuracy"] == maxAccuracy and stats["count"] > maxCount):
                    maxWord = word
                    maxAccuracy = stats["accuracy"]
                    maxCount = stats["count"]
            
            topWords.append(maxWord)
            del wordAccuracy[word]
        
        return topWords
    
    def __formatRegex(self, phrase):
        if self.gap in phrase:
            gapIndex = phrase.index(self.gap)
            gapStringLength = len(self.gap)
            gapLength = phrase[gapIndex + gapStringLength + 1:gapIndex + gapStringLength + 2]
            prefix = self.__formatRegex(phrase[0:gapIndex])
            suffix = self.__formatRegex(phrase[gapIndex + gapStringLength + 3])

            return prefix + self.gap + "{" + gapLength + "}" + suffix
        
        for token in self.regexTokens:
            phrase = phrase.replace(token, "\\" + token).replace("\\\\", "\\")
        
        return phrase
