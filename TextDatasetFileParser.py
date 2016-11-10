class Instance(object):
    def __init__(self, text, class_value, weight):
        self.text = text
        self.class_value = class_value
        self.weight = weight

class TextDatasetFileParser(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def parse(self, filename):
        file = open(filename, 'r')
        dataset = []

        if filename.endswith(".arff"):
            dataset = self.__parse_arff_file(file)
        elif filename.endswith(".csv"):
            dataset = self.__parse_csv_file(file)

        file.close()
        return dataset

    def __parse_arff_file(self, file):
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

                if (text.startswith("'") and text.endswith("'")) or (text.startswith('"')
                                                                     and text.endswith('"')):
                    text = text[1:len(text) - 1]

                dataset.append(Instance(text, class_value, weight))
            elif line == "\n":
                continue

        return dataset

    def __parse_csv_file(self, file):
        dataset = []

        while True:
            line = file.readline()

            if line == "":
                break
            elif line == "\n":
                continue
            else:
                line = line[0:len(line) - 1]

                if self.verbose:
                    print(line)

                last_comma_index = line.rfind(",")
                class_value = line[last_comma_index + 1:]
                text = line[0:last_comma_index]

                if (text.startswith("'") and text.endswith("'")) or (text.startswith('"')
                                                                     and text.endswith('"')):
                    text = text[1:len(text) - 1]

                if (class_value.startswith("'")
                    and class_value.endswith("'")) or (class_value.startswith('"')
                                                       and class_value.endswith('"')):
                    class_value = class_value[1:len(class_value) - 1]

                dataset.append(Instance(text, class_value, 1))

        return dataset
                