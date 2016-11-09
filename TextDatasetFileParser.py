class Instance(object):
    def __init__(self, text, classValue, weight):
        self.text = text
        self.classValue = classValue
        self.weight = weight

class TextDatasetFileParser(object):
    def __init__(self, verbose = False):
        self.verbose = verbose
    
    def parse(self, filename):
        file = open(filename, 'r')
        dataset = []
        
        if filename.endswith(".arff"):
            dataset = self.__parseARFFFile(file)
        elif filename.endswith(".csv"):
            dataset = self.__parseCSVFile(file)
        
        file.close()
        return dataset
    
    def __parseARFFFile(self, file):
        dataTag = "@DATA"
        parsingData = False
        dataset = []
        
        while True:
            line = file.readline()

            if line == "":
                break
            elif line.startswith(dataTag) and not parsingData:
                parsingData = True
            elif parsingData:
                line = line[0:len(line) - 1]
                            
                if self.verbose:
                    print(line)
                
                weight = 1
                lastCommaIndex = line.rfind(",")
                lastColumn = line[lastCommaIndex + 1:len(line)]
                
                if lastColumn.startswith("{") and lastColumn.endswith("}"):
                    weight = float(lastColumn[1:len(lastColumn) - 1])
                    line = line[0:lastCommaIndex]
                    lastCommaIndex = line.rfind(",")
                    lastColumn = line[lastCommaIndex + 1:len(line)]
                
                classValue = lastColumn
                text = line[0:lastCommaIndex]
                
                if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
                    text = text[1:len(text) - 1]
                
                dataset.append(Instance(text, classValue, weight))
            elif line == "\n":
                continue
        
        return dataset
    
    def __parseCSVFile(self, file):
        dataset = []
        
        while(True):
            line = file.readline()
            
            if line == "":
                break
            elif line == "\n":
                continue
            else:
                line = line[0:len(line) - 1]
                
                if self.verbose:
                    print(line)
                
                lastCommaIndex = line.rfind(",")
                classValue = line[lastCommaIndex + 1:len(line)]
                text = line[0:lastCommaIndex]
                
                if (text.startswith("'") and text.endswith("'")) or (text.startswith('"') and text.endswith('"')):
                    text = text[1:len(text) - 1]
                
                dataset.append(Instance(text, classValue, 1))
	return dataset
