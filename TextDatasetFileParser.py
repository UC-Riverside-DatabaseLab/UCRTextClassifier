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
        file = open(filename, 'r')
        parsing_data = False
        dataset = []

        while True:
            line = file.readline()

            if line == "":
                break
            elif line.startswith("@DATA") and not parsing_data:
                parsing_data = True
            elif parsing_data:
                line = line[0:len(line) - 1]

                if self.verbose:
                    print(line)

                weight = 1
                last_comma_index = line.rfind(",")
                last_column = line[last_comma_index + 1:]

                if last_column.startswith("{") and last_column.endswith("}"):
                    weight = float(last_column[1:len(last_column) - 1])
                    line = line[0:last_comma_index]
                    last_comma_index = line.rfind(",")
                    last_column = line[last_comma_index + 1:]

                class_value = last_column
                text = line[0:last_comma_index]

                if (text.startswith("'") and text.endswith("'") or
                        text.startswith('"') and text.endswith('"')):
                    text = text[1:len(text) - 1]

                dataset.append(Instance(text, class_value, weight))
            elif line == "\n":
                continue

        file.close()
        return dataset

    def __parse_csv_file(self, filename, delimit, quote_char):
        dataset = []

        with open(filename, newline="") as file:
            reader = csv.reader(file, delimiter=delimit, quotechar=quote_char)

            for line in reader:
                if self.verbose:
                    print(line)

                text = line[0]
                class_value = line[1]

                if len(line) > 2:
                    weight = float(line[2])
                else:
                    weight = 1

                dataset.append(Instance(text, class_value, weight))

        return dataset
