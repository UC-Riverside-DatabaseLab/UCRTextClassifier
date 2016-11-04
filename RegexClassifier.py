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
        self.classValue = classValue
        self.numCorrect = numCorrect
        self.score = score
    
    def matches(self, s):
        return self.regex.match(s)

class RegexClassifier(object):
    def __init__(self, scoringMethod = ScoringMethod.accuracy, scoreThreshold = 1, jumpLength = 2, rootWords = 1, minRootWordFrequency = "auto"):
        self.regexPrefix = "(^|^.* )"
        self.regexSuffix = "($| .*$)"
        self.gap = " (\\S+ )"
        self.matchedPattern = "MATCHED_PATTERN"
        self.regexTokens = [".", "^", "$", "(", ")", "[", "]", "{", "}", "?", "+", "|", "*"]
        self.jumpLength = max(0, jumpLength)
        self.scoreThreshold = scoreThreshold
        self.rootWords = max(1, rootWords)
        self.minRootWordFrequency = minRootWordFrequency
        self.scoringMethod = scoringMethod
    
    def train(self, data):
        self.regexRules = []
        prefixes = set()
        suffixes = set()
        currentRegexRules = []
        newRegexRules = []
        improved = False
        
        for word in self.__findTopWords(data):
            currentRegexRules.append(self.__createRegexRule(word, data))
            
            while len(currentRegexRules) > 0:
                newRegexRules.clear()

                for regexRule in currentRegexRules:
                    if regexRule == None or regexRule.phrase == None:
                        continue
                    
                    improved = False
                    candidates = []
                    
                    for i in range(0, self.jumpLength + 1):
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
            if regexRule.matches(instance.text):
                return regexRule.distribution
        
        return {}
    
    def setScoreThreshold(self, scoreThreshold):
        self.scoreThreshold = scoreThreshold
    
    def __createDefaultRegexRule(self, data):
        distribution = {}
        maxClass = None
        maxValue = 0
        total = 0
        
        for instance in data:
            matched = False
            
            for regexRule in self.regexRules:
                if regexRule.matches(instance.text):
                    matched = True

            if not matched:
                if instance.classValue in distribution:
                    distribution[instance.classValue] = distribution[instance.classValue] + instance.weight
                else:
                    distribution[instance.classValue] = instance.weight
                
                total = total + instance.weight
                
                if distribution[instance.classValue] > maxValue:
                    maxValue = distribution[instance.classValue]
                    maxClass = instance.classValue
        
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
        
        for instance in data:
            if regex.match(instance.text):
                total = total + instance.weight
                
                if instance.classValue in distribution:
                    distribution[instance.classValue] = distribution[instance.classValue] + instance.weight
                else:
                    distribution[instance.classValue] = instance.weight
                
                if distribution[instance.classValue] > numCorrect:
                    numCorrect = distribution[instance.classValue]
                    classValue = instance.classValue
                
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
            text = regexRule.partialRegex.sub(self.matchedPattern, instance.text)
            
            while self.matchedPattern in text:
                partialText = text[0:text.index(self.matchedPattern)].strip().split(" ")
                partialTextLength = len(partialText)
                
                if gapSize < partialTextLength:
                    prefixes.add(partialText[partialTextLength - 1 - gapSize])
                
                text = text[text.index(self.matchedPattern) + len(self.matchedPattern):len(text)].strip()
                partialText = text.split(" ")
                
                if gapSize < len(partialText):
                    suffixes.add(partialText[gapSize])
    
    def __findTopWords(self, data):
        topWords = []
        words = {}
        wordAccuracy = {}
        
        if self.minRootWordFrequency == "auto":
            minRootWordFrequency = len(data) / 2
        else:
            minRootWordFrequency = max(1, self.minRootWordFrequency)
        
        for instance in data:
            for word in instance.text.split(" "):
                if word == "":
                    continue
                
                if word in words:
                    if instance.classValue in words[word]:
                        words[word][instance.classValue] = words[word][instance.classValue] + instance.weight
                    else:
                        words[word][instance.classValue] = instance.weight
                else:
                    words[word] = {instance.classValue : instance.weight}
        
        for word, distribution in words.items():
            maxCount = 0
            total = 0
            
            for classValue, count in distribution.items():
                total = total + count
                
                if count > maxCount:
                    maxCount = count
            
            
            if total > minRootWordFrequency:
                wordAccuracy[word] = {"accuracy" : maxCount / total, "count" : maxCount}
        
        for i in range(0, self.rootWords):
            maxWord = None
            maxAccuracy = 0
            maxCount = 0
            
            for word, stats in wordAccuracy.items():
                if stats["accuracy"] > maxAccuracy or (stats["accuracy"] == maxAccuracy and stats["count"] > maxCount):
                    maxWord = word
                    maxAccuracy = stats["accuracy"]
                    maxCount = stats["count"]
            
            if maxWord != None:
                topWords.append(maxWord)
                del wordAccuracy[maxWord]
        
        return topWords
    
    def __formatRegex(self, phrase):
        if self.gap in phrase:
            gapIndex = phrase.index(self.gap)
            gapStringLength = len(self.gap)
            gapLength = phrase[gapIndex + gapStringLength + 1:gapIndex + gapStringLength + 2]
            prefix = self.__formatRegex(phrase[0:gapIndex])
            suffix = self.__formatRegex(phrase[gapIndex + gapStringLength + 3:len(phrase)])

            return prefix + self.gap + "{" + gapLength + "}" + suffix
        
        for token in self.regexTokens:
            phrase = phrase.replace(token, "\\" + token).replace("\\\\", "\\")
        
        return phrase
