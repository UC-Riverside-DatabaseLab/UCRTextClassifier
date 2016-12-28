import re
from enum import Enum
from AbstractTextClassifier import AbstractTextClassifier


class ScoringMethod(Enum):
    """Enumeration of regular expression scoring methods. Information gain has
    not yet been implemented.
    """
    accuracy = 1
    information_gain = 2


class RegexRule(object):
    """Container for a regular expression, its associated class, etc.

    Constructor arguments:
    regex - The regular expression
    phrase - The phrase the regular expression was based on
    matched - A list of instance objects (defined in TextDataSetFileParser.py)
        whose text is matched by the regular expression
    distribution - The class distribution among matched instances
    class_value - The class that matched instances will be classified as
    num_correct - The number of classified instances in matched of class
        class_value
    score - Accuracy or information gain, according to the scoring method
    """
    def __init__(self, regex, phrase, matched, distribution, class_value,
                 num_correct, score):
        self.regex = regex
        pattern = regex.pattern.replace("(^|^.* )", "").replace("($| .*$)", "")
        self.partial_regex = re.compile(pattern)
        self.phrase = phrase
        self.matched = matched
        self.distribution = distribution
        self.class_value = class_value
        self.num_correct = num_correct
        self.score = score

    def matches(self, text):
        """Determine whether the regular expression matches the given text.

        Arguments:
        text - The text to compare to the regular expression

        Returns:
        True if the regular expression matched the given text, False otherwise.
        """
        return self.regex.match(text)


class RegexClassifier(AbstractTextClassifier):
    """Classifier that generates regular expressions based on training data.

    Constructor arguments:
    scoring_method (default accuracy) - The scoring method to use while
        expanding regular expressions
    score_threshold (default 1) - The minimum score of maximally-expanded
        regular expressions to accept for classification
    jump_length (default 2) - The maximum number of consecutive nearby words to
        skip when expanding a regular expression
    root_words (default 1) - The number of initial words to expand regular
        expressions from
    min_root_word_frequency (default 0.25) - The minimum frequency of words (as
        a percentage of the sum of the dataset's weights) to consider as root
        words
    """
    def __init__(self, scoring_method=ScoringMethod.accuracy,
                 score_threshold=1, jump_length=2, root_words=1,
                 min_root_word_frequency=0.25):
        self.regex_prefix = "(^|^.* )"
        self.regex_suffix = "($| .*$)"
        self.gap = " (\\S+ )"
        self.regex_tokens = [".", "^", "$", "(", ")", "[", "]", "{", "}", "?",
                             "+", "|", "*"]
        self.jump_length = max(0, jump_length)
        self.score_threshold = score_threshold
        self.root_words = max(1, root_words)
        self.min_root_word_frequency = min(1, max(0, min_root_word_frequency))
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
                        self.__find_prefixes_and_suffixes(regex_rule,
                                                          regex_rule.matched,
                                                          prefixes, suffixes,
                                                          i)
                        self.__expand_regex(regex_rule.phrase, candidates,
                                            prefixes, regex_rule.matched, True,
                                            i)
                        self.__expand_regex(regex_rule.phrase, candidates,
                                            suffixes, regex_rule.matched,
                                            False, i)

                    for new_regex_rule in candidates:
                        new_score = new_regex_rule.score
                        score = regex_rule.score

                        if (new_score > score or new_score == score and
                                new_regex_rule.num_correct >
                                regex_rule.num_correct):
                            new_regex_rules.append(new_regex_rule)

                            improved = True

                    if (not improved and regex_rule.score >=
                            self.score_threshold):
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

    def __create_default_regex_rule(self, data):
        distribution = {}
        max_class = None
        max_value = 0
        total = 0

        for instance in data:
            matched = False

            for regex_rule in self.regex_rules:
                if regex_rule.matches(instance.text):
                    matched = True

            if not matched:
                if instance.class_value in distribution:
                    distribution[instance.class_value] += instance.weight
                else:
                    distribution[instance.class_value] = instance.weight

                total += instance.weight

                if distribution[instance.class_value] > max_value:
                    max_value = distribution[instance.class_value]
                    max_class = instance.class_value

        for c_value, count in distribution.items():
            distribution[c_value] = count / total

        self.regex_rules.append(RegexRule(re.compile(".*"), ".*", None,
                                          distribution, max_class, max_value,
                                          distribution[max_class]))

    def __create_regex_rule(self, phrase, data):
        distribution = {}
        total = 0
        num_correct = 0
        class_value = None
        regex = re.compile(self.regex_prefix + self.__format_regex(phrase) +
                           self.regex_suffix)
        matched = []

        for instance in data:
            if regex.match(instance.text):
                total += instance.weight

                if instance.class_value in distribution:
                    distribution[instance.class_value] += instance.weight
                else:
                    distribution[instance.class_value] = instance.weight

                if distribution[instance.class_value] > num_correct:
                    num_correct = distribution[instance.class_value]
                    class_value = instance.class_value

                matched.append(instance)

        if total > 0:
            for c_value, count in distribution.items():
                distribution[c_value] = count / total

            return RegexRule(regex, phrase, matched, distribution, class_value,
                             num_correct, distribution[class_value])

        return None

    def __expand_regex(self, phrase, candidates, affixes, data, use_prefixes,
                       gap_size):
        formatted_gap = " "
        regex_rule = None

        if gap_size > 0:
            formatted_gap = self.gap + "{" + str(gap_size) + "}"

        for affix in affixes:
            if use_prefixes:
                formatted_phrase = affix + formatted_gap + phrase
            else:
                formatted_phrase = phrase + formatted_gap + affix

            regex_rule = self.__create_regex_rule(formatted_phrase, data)

            if regex_rule is not None:
                candidates.append(regex_rule)

        return candidates

    def __find_prefixes_and_suffixes(self, regex_rule, data, prefixes,
                                     suffixes, gap_size):
        prefixes.clear()
        suffixes.clear()

        matched_pattern = "MATCHED_PATTERN"
        matched_len = len(matched_pattern)

        for instance in data:
            text = regex_rule.partial_regex.sub(matched_pattern, instance.text)

            while matched_pattern in text:
                matched_pattern_index = text.index(matched_pattern)
                partial_text = text[0:matched_pattern_index].strip().split(" ")
                partial_text_length = len(partial_text)

                if gap_size < partial_text_length:
                    prefix_index = partial_text_length - 1 - gap_size
                    prefixes.add(partial_text[prefix_index])

                text = text[text.index(matched_pattern) + matched_len:].strip()
                partial_text = text.split(" ")

                if gap_size < len(partial_text):
                    suffixes.add(partial_text[gap_size])

    def __find_top_words(self, data):
        top_words = []
        words = {}
        word_accuracy = {}
        sum_of_weights = 0
        acc = "accuracy"
        cnt = "count"

        for instance in data:
            sum_of_weights += instance.weight

            for word in instance.text.split(" "):
                if word == "":
                    continue

                if word in words:
                    if instance.class_value in words[word]:
                        words[word][instance.class_value] += instance.weight
                    else:
                        words[word][instance.class_value] = instance.weight
                else:
                    words[word] = {instance.class_value: instance.weight}

        for word, distribution in words.items():
            max_count = 0
            total = 0

            for class_value, count in distribution.items():
                total += count

                if count > max_count:
                    max_count = count

            if total > sum_of_weights * self.min_root_word_frequency:
                word_accuracy[word] = {acc: max_count / total, cnt: max_count}

        for i in range(0, self.root_words):
            max_word = None
            max_accuracy = 0
            max_count = 0

            for word, stats in word_accuracy.items():
                if (stats[acc] > max_accuracy or stats[acc] == max_accuracy and
                        stats[cnt] > max_count):
                    max_word = word
                    max_accuracy = stats[acc]
                    max_count = stats[cnt]

            if max_word is not None:
                top_words.append(max_word)
                del word_accuracy[max_word]

        return top_words

    def __format_regex(self, phrase):
        if self.gap in phrase:
            gap_index = phrase.index(self.gap)
            gap_string_length = len(self.gap)
            start = gap_index + gap_string_length + 1
            gap_length = phrase[start:gap_index + gap_string_length + 2]
            prefix = self.__format_regex(phrase[0:gap_index])
            start = gap_index + gap_string_length + 3
            suffix = self.__format_regex(phrase[start:])

            return prefix + self.gap + "{" + gap_length + "}" + suffix

        for token in self.regex_tokens:
            phrase = phrase.replace(token, "\\" + token).replace("\\\\", "\\")

        return phrase
