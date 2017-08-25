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
        self.verbose = verbose

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
            reader = csv.reader(file, quotechar="'")
            parsing_data = False
            attributes = []

            for line in reader:
                length = len(line)

                if not parsing_data and length > 0 and \
                        line[0].upper() == "@DATA":
                    parsing_data = True
                elif not parsing_data and length > 0 and \
                        line[0].upper().startswith("@ATTRIBUTE"):
                    if length > 1 or line[0].find("{") > 0:
                        data_type = "nominal"
                    else:
                        data_type = line[0][line[0].rfind(" ") + 1:].lower()

                    attributes.append(data_type)

                elif parsing_data and length > 0:
                    text = ""
                    weight = 1
                    values = []

                    if self.verbose:
                        print(line)

                    if length > 2:
                        weighted = False

                        if line[length - 1].startswith("{") and \
                                line[length - 1].endswith("}"):
                            weight = line[length - 1]
                            weight = float(weight[1:len(weight) - 1])
                            weighted = True

                        for i in range(0, length - (2 if weighted else 1)):
                            attribute = attributes[i]
                            value = line[i]

                            if attribute == "string" or attribute == "nominal":
                                text += (" " if len(text) > 0 else "") + value
                            else:
                                values.append(float(value))

                        label = line[length - (2 if weighted else 1)]

                    dataset.append(Instance(text, label, weight, values))

        return dataset

    def __parse_csv_file(self, filename, delimit, quote_char):
        dataset = []

        with open(filename, newline="", errors="ignore") as file:
            reader = csv.reader(file, delimiter=delimit, quotechar=quote_char)

            for line in reader:
                if self.verbose:
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
