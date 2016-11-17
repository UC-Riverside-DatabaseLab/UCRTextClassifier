import csv


class Instance(object):
    def __init__(self, text, class_value, weight):
        self.text = text
        self.class_value = class_value
        self.weight = weight


class TextDatasetFileParser(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def parse(self, filename):
        if filename.endswith(".arff"):
            return self.__parse_arff_file(filename)
        elif filename.endswith(".csv"):
            return self.__parse_csv_file(filename)

        return []

    def parse_unlabeled(self, filename):
        dataset = []

        with open(filename, newline="") as file:
            reader = csv.reader(file)

            for line in reader:
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

    def __parse_csv_file(self, filename):
        dataset = []

        with open(filename, newline="") as file:
            reader = csv.reader(file)

            for line in reader:
                text = line[0]
                class_value = line[1]

                if len(line) > 2:
                    weight = float(line[2])
                else:
                    weight = 1

                dataset.append(Instance(text, class_value, weight))

        return dataset
