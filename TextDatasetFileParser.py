import csv


class Instance(object):
    """Container for a single text data instance.

    Constructor arguments:
    text - Text data string
    class_value - The class of the text data
    weight (default 1) - The weight of the text data
    values (default []) - A list of non-text values
    """
    def __init__(self, text, class_value, weight=1, values=[]):
        self.text = text
        self.class_value = class_value
        self.weight = weight
        self.values = values


class TextDatasetFileParser(object):
    """Reader for text dataset files.

    Constructor arguments:
    verbose (default False) - If True, print each line of the file as it's read
    """
    def __init__(self, verbose=False):
        self.__verbose = verbose

    def parse(self, filename, delimiter=",", quotechar='"'):
        """Read an ARFF or CSV file containing a dataset. ARFF files should be
        formatted according to Weka's standards. CSV files should be
        in "text,class,weight" format (weight is optional).

        Arguments:
        filename - The path of the file to read

        Returns:
        A list of Instance objects made from the data contained in the file
        """
        if filename.endswith(".arff"):
            return self.__parse_arff_file(filename)
        elif filename.endswith(".csv"):
            return self.__parse_csv_file(filename, delimiter, quotechar)

        return []

    def parse_unlabeled(self, filename):
        """Read a text file containing unlabeled text data. The file should
        have one text data string per line.

        Arguments:
        filename - The path of the file to read

        Returns:
        A list of strings made from the data contained in the file
        """
        dataset = []

        with open(filename, newline="", errors="ignore") as file:
            for line in file:
                dataset.append(line.lower())

        return dataset

    def __parse_arff_file(self, filename):
        dataset = []

        with open(filename, newline="", errors="ignore") as file:
            parsing_data = False
            attributes = []

            for line in file:
                line = line.strip()

                if len(line) == 0:
                    continue
                elif not parsing_data and \
                        line.upper().startswith("@ATTRIBUTE"):
                    if line.find("{") >= 0:
                        data_type = "NOMINAL"
                    else:
                        data_type = line[line.rfind(" ") + 1:].upper()

                    if self.__verbose:
                        print("Attribute: " + data_type)

                    attributes.append(data_type)
                elif not parsing_data and line.upper() == "@DATA":
                    parsing_data = True
                elif parsing_data:
                    current_attribute = 0
                    text = ""
                    value = ""
                    values = []
                    weight = 1
                    in_quotes = False

                    if self.__verbose:
                        print(line)

                    if line.endswith("}"):
                        index = line.rfind(",{")

                        if index >= 0:
                            weight = float(line[index + 2:len(line) - 1])
                            line = line[:index]

                    index = line.rfind(",")
                    label = line[index + 1:]
                    line = line[:index]

                    for i in range(0, len(line)):
                        if line[i] == "'" and (i == 0 or line[i - 1] != "\\"):
                            in_quotes = not in_quotes
                        elif not in_quotes and line[i] == ",":
                            if attributes[current_attribute] == "STRING" or \
                                    attributes[current_attribute] == "NOMINAL":
                                text += (" " if len(text) > 0 else "") + value
                            else:
                                values.append(float(value))

                            value = ""
                            current_attribute += 1
                        elif line[i] != "\\":
                            value += line[i]

                    if attributes[current_attribute] == "STRING" or \
                            attributes[current_attribute] == "NOMINAL":
                        text += (" " if len(text) > 0 else "") + value
                    else:
                        values.append(float(value))

                    dataset.append(Instance(text, label, weight, values))

        return dataset

    def __parse_csv_file(self, filename, delimit, quote_char):
        dataset = []

        with open(filename, newline="", errors="ignore") as file:
            reader = csv.reader(file, delimiter=delimit, quotechar=quote_char)

            for line in reader:
                if self.__verbose:
                    print(line)

                if len(line) > 1:
                    class_value = line[1]
                else:
                    class_value = None

                if len(line) > 2:
                    weight = float(line[2])
                else:
                    weight = 1

                dataset.append(Instance(line[0], class_value, weight))

        return dataset
