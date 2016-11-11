import re
from enum import Enum
from AbstractTextClassifier import AbstractTextClassifier

class ScoringMethod(Enum):
    accuracy = 1
    informationGain = 2

class RegexRule(object):
    def __init__(self, regex, phrase, matched, distribution, class_value, num_correct, score):
        self.regex = regex
        self.partial_regex = re.compile(regex.pattern.replace("(^|^.* )", "").replace("($| .*$)",
                                                                                      ""))
        self.phrase = phrase
        self.matched = matched
        self.distribution = distribution
        self.class_value = class_value
        self.num_correct = num_correct
        self.score = score

    def matches(self, text):
        return self.regex.match(text)

class RegexClassifier(AbstractTextClassifier):
    def __init__(self, scoring_method=ScoringMethod.accuracy, score_threshold=1, jump_length=2,
                 root_words=1, min_root_word_frequency="auto"):
        self.regex_prefix = "(^|^.* )"
        self.regex_suffix = "($| .*$)"
        self.gap = " (\\S+ )"
        self.matched_pattern = "MATCHED_PATTERN"
        self.regex_tokens = [".", "^", "$", "(", ")", "[", "]", "{", "}", "?", "+", "|", "*"]
        self.jump_length = max(0, jump_length)
        self.score_threshold = score_threshold
        self.root_words = max(1, root_words)
        self.min_root_word_frequency = min_root_word_frequency
        self.scoring_method = scoring_method
        self.regex_rules = []

    def train(self, data):
        self.regex_rules = []
        prefixes = set()
        suffixes = set()
        current_regex_rules = []
        new_regex_rules = []
        improved = False

        for word in self.__find_top_words(data):
            current_regex_rules.append(self.__create_regex_rule(word, data))

            while len(current_regex_rules) > 0:
                new_regex_rules.clear()

                for regex_rule in current_regex_rules:
                    if regex_rule is None or regex_rule.phrase is None:
                        continue

                    improved = False
                    candidates = []

                    for i in range(0, self.jump_length + 1):
                        self.__find_prefixes_and_suffixes(regex_rule, regex_rule.matched, prefixes,
                                                          suffixes, i)
                        self.__expand_regex(regex_rule.phrase, candidates, prefixes,
                                            regex_rule.matched, True, i)
                        self.__expand_regex(regex_rule.phrase, candidates, suffixes,
                                            regex_rule.matched, False, i)

                    for new_regex_rule in candidates:
                        new_score = new_regex_rule.score
                        score = regex_rule.score

                        if new_score > score or (new_score == score and new_regex_rule.num_correct
                                                 > regex_rule.num_correct):
                            new_regex_rules.append(new_regex_rule)

                            improved = True

                    if not improved and regex_rule.score >= self.score_threshold:
                        self.regex_rules.append(regex_rule)

                current_regex_rules = new_regex_rules.copy()

        for regex_rule in self.regex_rules:
            regex_rule.matched = None

        self.__create_default_regex_rule(data)

    def classify(self, instance):
        for regex_rule in self.regex_rules:
            if regex_rule.matches(instance.text):
                return regex_rule.distribution

        return {}

    def set_score_threshold(self, score_threshold):
        self.score_threshold = score_threshold

    def __create_default_regex_rule(self, data):
        distribution = {}
        max_class = None
        max_value = 0
        total = 0

        for instance in data:
            matched = False
            inst_class = instance.class_value

            for regex_rule in self.regex_rules:
                if regex_rule.matches(instance.text):
                    matched = True

            if not matched:
                if instance.class_value in distribution:
                    distribution[inst_class] = distribution[inst_class] + instance.weight
                else:
                    distribution[inst_class] = instance.weight

                total = total + instance.weight

                if distribution[inst_class] > max_value:
                    max_value = distribution[inst_class]
                    max_class = inst_class

        for c_value, count in distribution.items():
            distribution[c_value] = count / total

        self.regex_rules.append(RegexRule(re.compile(".*"), ".*", None, distribution, max_class,
                                          max_value, distribution[max_class]))

    def __create_regex_rule(self, phrase, data):
        distribution = {}
        total = 0
        num_correct = 0
        class_value = None
        regex = re.compile(self.regex_prefix + self.__format_regex(phrase) + self.regex_suffix)
        matched = []

        for instance in data:
            inst_class = instance.class_value

            if regex.match(instance.text):
                total = total + instance.weight

                if inst_class in distribution:
                    distribution[inst_class] = distribution[inst_class] + instance.weight
                else:
                    distribution[inst_class] = instance.weight

                if distribution[inst_class] > num_correct:
                    num_correct = distribution[inst_class]
                    class_value = inst_class

                matched.append(instance)

        if total > 0:
            for c_value, count in distribution.items():
                distribution[c_value] = count / total

            return RegexRule(regex, phrase, matched, distribution, class_value, num_correct,
                             distribution[class_value])

        return None

    def __expand_regex(self, phrase, candidates, affixes, data, use_prefixes, gap_size):
        formatted_gap = " "
        regex_rule = None

        if gap_size > 0:
            formatted_gap = self.gap + "{" + str(gap_size) + "}"

        for affix in affixes:
            if use_prefixes:
                regex_rule = self.__create_regex_rule(affix + formatted_gap + phrase, data)
            else:
                regex_rule = self.__create_regex_rule(phrase + formatted_gap + affix, data)

            if regex_rule != None:
                candidates.append(regex_rule)

        return candidates

    def __find_prefixes_and_suffixes(self, regex_rule, data, prefixes, suffixes, gap_size):
        prefixes.clear()
        suffixes.clear()

        for instance in data:
            text = regex_rule.partial_regex.sub(self.matched_pattern, instance.text)

            while self.matched_pattern in text:
                partial_text = text[0:text.index(self.matched_pattern)].strip().split(" ")
                partial_text_length = len(partial_text)

                if gap_size < partial_text_length:
                    prefixes.add(partial_text[partial_text_length - 1 - gap_size])

                text = text[text.index(self.matched_pattern) + len(self.matched_pattern):].strip()
                partial_text = text.split(" ")

                if gap_size < len(partial_text):
                    suffixes.add(partial_text[gap_size])

    def __find_top_words(self, data):
        top_words = []
        words = {}
        word_accuracy = {}

        if self.min_root_word_frequency == "auto":
            min_root_word_frequency = len(data) / 2
        else:
            min_root_word_frequency = max(1, self.min_root_word_frequency)

        for instance in data:
            inst_class = instance.class_value

            for word in instance.text.split(" "):
                if word == "":
                    continue

                if word in words:
                    if inst_class in words[word]:
                        words[word][inst_class] = words[word][inst_class] + instance.weight
                    else:
                        words[word][inst_class] = instance.weight
                else:
                    words[word] = {inst_class : instance.weight}

        for word, distribution in words.items():
            max_count = 0
            total = 0

            for class_value, count in distribution.items():
                total = total + count

                if count > max_count:
                    max_count = count

            if total > min_root_word_frequency:
                word_accuracy[word] = {"accuracy" : max_count / total, "count" : max_count}

        for i in range(0, self.root_words):
            max_word = None
            max_accuracy = 0
            max_count = 0

            for word, stats in word_accuracy.items():
                if stats["accuracy"] > max_accuracy or (stats["accuracy"] == max_accuracy
                                                        and stats["count"] > max_count):
                    max_word = word
                    max_accuracy = stats["accuracy"]
                    max_count = stats["count"]

            if max_word != None:
                top_words.append(max_word)
                del word_accuracy[max_word]

        return top_words

    def __format_regex(self, phrase):
        if self.gap in phrase:
            gap_index = phrase.index(self.gap)
            gap_string_length = len(self.gap)
            gap_length = phrase[gap_index + gap_string_length + 1:gap_index + gap_string_length + 2]
            prefix = self.__format_regex(phrase[0:gap_index])
            suffix = self.__format_regex(phrase[gap_index + gap_string_length + 3:])

            return prefix + self.gap + "{" + gap_length + "}" + suffix

        for token in self.regex_tokens:
            phrase = phrase.replace(token, "\\" + token).replace("\\\\", "\\")

        return phrase
