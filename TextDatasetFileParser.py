import csv


class Instance(object):
    """Container for a single text data instance.

    Constructor arguments:
    text - Text data string
    class_value - The class of the text data
    weight - The weight of the text data
    """
    def __init__(self, text, class_value, weight=1):
        self.text = text
        self.class_value = class_value
        self.weight = weight


class TextDatasetFileParser(object):
    """Reader for text dataset files.

    Constructor arguments:
    merge_extra_columns (default False) - If True, columns between the first
    and the class attribute column in .arff files are concatenated to the first
    column, separated by a comma. Otherwise, they're ignored.
    verbose (default False) - If True, print each line of the file as it's read
    """
    def __init__(self, merge_extra_columns=False, verbose=False):
        self.merge_extra_columns = merge_extra_columns
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
        parsing_data = False
        dataset = []

        with open(filename, newline="", errors="ignore") as file:
            reader = csv.reader(file, quotechar="'")

            for line in reader:
                length = len(line)
                weight = 1

                if not parsing_data and length > 0 and \
                        line[0].upper() == "@DATA":
                    parsing_data = True
                elif parsing_data and length > 0:
                    if self.verbose:
                        print(line)

                    if length > 2:
                        weighted = False

                        if line[length - 1].startswith("{") and \
                                line[length - 1].endswith("}"):
                            weight = float(line[2][1:len(line[2]) - 1])
                            weighted = True

                        if self.merge_extra_columns:
                            for i in range(1, length - (2 if weighted else 1)):
                                line[0] += "," + line[i]

                            line[1] = line[length - (2 if weighted else 1)]

                    dataset.append(Instance(line[0], line[1], weight))

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
