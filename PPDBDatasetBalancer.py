from mysql import connector
from nltk import word_tokenize


class PPDBDatasetBalancer(object):
    def __init__(self, scoring_feature, threshold=0, input_range=(1, 1),
                 host, database, user, password):
        self.__scoring_feature = scoring_feature
        self.__threshold = threshold
        self.__input_range = input_range
        self.__connection = connector.connect(host=host, database=database,
                                              user=user, password=password)

        self.__input_range[0] = max(1, self.__input_range[0])
        self.__input_range[1] = max(1, self.__input_range[1])

        if self.__input_range[0] > self.__input_range[1]:
            self.__input_range = self.__input_range[::-1]

    def balance(self, data):
        if not self.__connection.is_connected():
            print("Not connected to database.")
            return

        new_instances = []

        for instance in data:
            class_value = instance.class_value
            words = word_tokenize(instance.text)
            num_words = len(words)
            paraphrases = {}

            for i in range(0, num_words):
                min_cutoff = min(num_words, i + self.__input_range[0])
                max_cutoff = min(num_words, i + self.__input_range[1])
                phrase = ""
                length = 0

                for j in range(i, min_cutoff):
                    phrase += (" " if length > 0 else "") + words[j]
                    length += 1

                if length < input_range[0]:
                    break

                self.__find_paraphrases(phrase, paraphrases)

                for j in range(i + length, max_cutoff):
                    phrase += (" " if length > 0 else "") + words[j]
                    length += 1

                    self.__find_paraphrases(phrase, paraphrases)

            for phrase in paraphrases.keys():
                index = instance.text.find(phrase)
                text = instance.text.replace(phrase, "")

                for paraphrase in paraphrases[phrase].keys():
                    paraphrased = text[0:index] + paraphrase + text[index:]

                    new_instances.append(Instance(paraphrased, class_value))

        return data + new_instances

    def close_connection(self):
        self.__connection.close()

    def __find_paraphrases(self, phrase, paraphrases):
        sql = "SELECT paraphrase, PPDB2Score FROM ppdb WHERE phrase = '"
        sql += phrase.replace("\\", "\\\\").replace("'", "\\'")
        sql += "' AND PPDB2Score > " + str(self.__threshold)
        cursor = self.__connection.cursor()

        cursor.execute(sql)

        for row in cursor.fetchall():
            if phrase not in paraphrases:
                paraphrases[phrase] = {}

            paraphrases[phrase][row[0]] = row[1]

        cursor.close()
