import csv


class Instance(object):
    """Container for a single text data instance.

    Constructor arguments:
    text - Text data string
    class_value - The class of the text data
    weight - The weight of the text data
    """
    def __init__(self, text, class_value, weight):
        self.text = text
        self.class_value = class_value
        self.weight = weight


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

    def parse_unlabeled(self, filename, delimit=",", quote_char='"'):
        """Read a CSV file containing unlabeled text data. The CSV file should
        have one text data string per line.

        Arguments:
        filename - The path of the file to read

        Returns:
        A list of strings made from the data contained in the file
        """
        dataset = []

        with open(filename, newline="") as file:
            reader = csv.reader(file, delimiter=delimit, quotechar=quote_char)

            for line in reader:
                if self.verbose:
                    print(line[0])

                dataset.append(line[0])

        return dataset

    def __parse_arff_file(self, filename):
        parsing_data = False
        dataset = []

        with open(filename, newline="", errors="ignore") as file:
            reader = csv.reader(file, quotechar="'")

            for line in reader:
                if not parsing_data and len(line) > 0 and line[0] == "@DATA":
                    parsing_data = True
                elif parsing_data:
                    if self.verbose:
                        print(line)

                    if len(line) > 2:
                        weight = float(line[2][1:len(line[2]) - 1])
                    else:
                        weight = 1

                    dataset.append(Instance(line[0], line[1], weight))

        return dataset

    def __parse_csv_file(self, filename, delimit, quote_char):
        dataset = []

        with open(filename, newline="") as file:
            reader = csv.reader(file, delimiter=delimit, quotechar=quote_char)

            for line in reader:
                if self.verbose:
                    print(line)

                if len(line) > 2:
                    weight = float(line[2])
                else:
                    weight = 1

                dataset.append(Instance(line[0], line[1], weight))

        return dataset
