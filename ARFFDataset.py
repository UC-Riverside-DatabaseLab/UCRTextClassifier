class Instance(object):
    def __init__(self, values, weight):
        self.values = values
        self.weight = weight

class ARFFDataset(object):
        def __init__(self, filename):
            file = open(filename, 'r')
            self.classIndex = -1
            self.numClasses = 0
            self.attributes = []
            self.instances = []
            relationTag = "@RELATION "
            attributeTag = "@ATTRIBUTE "
            dataTag = "@DATA"
            parsingData = False
            
            while True:
                line = file.readline()

                if line == "":
                    break
                elif line.startswith(relationTag) and not parsingData:
                    self.relation = line.replace(relationTag, "")
                elif line.startswith(attributeTag) and not parsingData:
                    self.attributes.append(line.replace(attributeTag, "").split(" ")[0])
                elif line.startswith(dataTag) and not parsingData:
                    parsingData = True
                elif parsingData:
                    values = []
                    weight = 1
                    index = 0;
                    buildingString = False
                    
                    for value in line.split(","):
                        stringValue = str(value)
                        if stringValue.startswith("'") and not stringValue.endswith("'"):
                            values.append(stringValue)
                            buildingString = True
                        elif buildingString:
                            if stringValue.endsWith("'"):
                                stringValue = stringValue[0:len(stringValue) - 1]
                                
                                values[index].append(stringValue)
                                
                                index = index + 1
                                buildingString = False
                            else:
                                values[index].append(stringValue)
                        elif stringValue.startswith("{") and stringValue.endswith("}") and not buildingString:
                            weight = float(stringValue[1:len(stringValue - 1)])
                        else:
                            values.append(stringValue)
                            
                            index = index + 1

                    self.instances.append(Instance(values, weight))
                elif line == "\n":
                    continue
        
            file.close()
        
        def setClassAttribute(self, classIndex):
            classValues = set()
            
            for instance in self.instances:
                classValues.add(instance.values[classIndex])
            
            self.numClasses = len(classValues)
            self.classIndex = classIndex
