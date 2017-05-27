class Node(object):
    def __init__(self, node_val):
        self.node_val = node_val
        self.children = []
        self.parents = []

    def add_child(self, rel_child):
        rel, child = rel_child
        self.children += [rel_child]  # tuples of the form (rel, node)
        child.parents += [(rel, self)]   # tuples of the form (rel, node)

    def clone(self):
        node_val_clone = self.node_val
        n = Node(node_val_clone)

        for rel, child in self.children:
            cloned_child = child.clone()
            cloned_child.parents += [(rel, n)]
            n.children += [(rel, cloned_child)]

        return n

    def __repr__(self):
        word1, pos = self.node_val
        word = {} if word1 == "*" else "{word:/%s.*/}" % word1

        if len(self.children) == 0:
            return word

        child_str = ""

        for rel, child in self.children:
            if rel == "indir":
                child_str += " >> %s" % child
            else:
                child_str += " > %s" % child if rel != "neg" else " >neg {}"

        return "(%s %s)" % (word, child_str)

    def find_lemma(self, lemma):  # searches this node and all its children
        word, po = self.node_val

        if lemma == word[:len(lemma)]:
            return self

        for rel, child in self.children:
            matched_node = child.find_lemma(lemma)

            if matched_node:
                return matched_node

        return None

    def get_all_important_word_nodes(self, word_nodes):
        word, po = self.node_val
        is_neg_node = False

        if len(self.parents) > 0:  # is not root node
            rel, parent = self.parents[0]

            if rel == "neg":
                is_neg_node = True

        if word != "*" and not is_neg_node:
            # then this is an important word and we add it to the dict
            if word not in word_nodes:
                word_nodes[word] = []

            word_nodes[word] += [self]

        for rel, child in self.children:
            # regardless of whether this node is important
            # or not, we consider its children
            child.get_all_important_word_nodes(word_nodes)

    def find_any_lemma(self, list_of_lemma):
        # searches this node and all its children
        word, po = self.node_val
        for lemma in list_of_lemma:
            if lemma == word[:len(lemma)]:
                return self

        for rel, child in self.children:
            matched_node = child.find_anyemma(list_of_lemma)

            if matched_node:
                return matched_node

        return None

    def get_all_patterns(self):
        word, pos = self.node_val
        patterns = []

        # parents-----------------------
        if len(self.parents) > 0:  # parent1
            rel1, parent1 = self.parents[0]
            word_par1, pos_par1 = parent1.node_val
            patterns += ["* >neg %s" % word] if rel1 == "neg" else \
                ["%s > %s" % (word_par1, word)]

            if len(parent1.parents) > 0:  # parent2
                rel2, parent2 = parent1.parents[0]
                word_par2, pos_par2 = parent2.node_val
                patterns += ["* >>neg %s" % word] if rel2 == "neg" else \
                    ["%s >> %s" % (word_par2, word)]

        # 1 direct child-----------------------
        if len(self.children) > 0:  # child1
            for i in range(0, len(self.children)):
                p = self.create_pattern_for_direct_child(i, self)
                patterns += [p]

        # 2 direct children-----------------------
        if len(self.children) > 1:  # child1
            # first choose childIndices to combine
            for comb in self.get_index_comb(len(self.children)):
                p = self.create_pattern_for_direct_child(comb, self)
                patterns += [p]

        return patterns

    def change_word_to_star(self):
        self.node_val = ("*", "*")
        return self

    def create_pattern_for_direct_child(self, i):
        word, pos = self.node_val
        rel, child1 = self.children[i]
        word_ch1, pos_ch1 = child1.node_val
        return "%s >neg *" % word if rel == "neg" else \
            "%s > %s" % (word, word_ch1)

    def get_index_comb(self, n):
        indices = []

        for i in range(0, n):
            for j in range(i + 1, n):
                indices += [(i, j)]

                for k in range(j + 1, n):
                    indices += [(i, j, k)]

        return indices


class Tree(object):
    def __init__(self, root):
        self.root = root

    def __repr__(self):
        return self.root.__repr__()

    def find_lemma(self, lemma):
        return self.root.find_lemma(lemma)


class PatternExtractor(object):
    def __init__(self):
        pass

    def prune_node(self, node, tree, important_words):
        found_important_word = False

        for lemma in important_words:
            n = tree.find_lemma(lemma)

            if n:
                found_important_word = True
                break

        if not found_important_word and len(node.parents) == 0:
            return None

    def is_important_node(self, node, important_words):
        if len(node.parents) > 0:
            rel, parent = node.parents[0]

            if rel == "neg":
                return True

        word, po = node.node_val

        for lemma in important_words:
            if lemma == word[:len(lemma)]:
                node.node_val = (lemma, po)  # replace word for lemma in node
                return True

        return False

    def get_sub_tree(self, node, important_words):
        if len(node.children) > 0:
            good_children = []

            for rel, child in node.children:
                good_child = self.get_sub_tree(child, important_words)

                if good_child:
                    good_children += [(rel, child)]

            node.children = good_children

        if len(node.children) == 0:
            if not self.is_important_node(node, important_words):
                # doesn't have important children and is not important itself
                return None

            # doesn't have important children, but is important itself
            return node

        if not self.is_important_node(node, important_words):
            # has important children but is not important itself
            return node.change_word_to_star()

        return node  # has important children and is important itself

    def is_star(self, node):
        word, pos = node.node_val

        if word == "*" and pos == "*":
            return True

        return False

    def prune_empty_roots(self, node):
        while self.is_star(node) and len(node.children) == 1:
            rel, child = node.children[0]
            node = child

        return node

    # if a star node s with parent p has only one child c,
    # make c a grandchild of p with indirect relation
    def make_indir_relations(self, n):
        new_children = []

        for rel, child in n.children:
            was_replaced = False

            while self.is_star(child) and len(child.children) == 1:
                (rel, grand_child) = child.children[0]
                child = grand_child
                was_replaced = True

            if was_replaced:
                rel = "indir"
            new_children += [(rel, child)]
        n.children = new_children
        for rel, child in n.children:
            self.make_indir_relations(child)

    def create_pattern_for_direct_child(self, i, node):
        word, pos = node.node_val
        rel, child1 = node.children[i]
        word_ch1, pos_ch1 = child1.node_val
        return "%s >neg *" % word if rel == "neg" else \
            "%s > %s" % (word, word_ch1)

    def create_pattern_for_direct_children(self, index_comb, node):
        word, pos = node.node_val
        child_patt = ""
        for index in index_comb:
            rel, child = node.children[index]
            word_ch, pos_ch = child.node_val
            child_patt += " >neg *" if rel == "neg" else " > %s" % word_ch

        return "%s %s" % (word, child_patt)

    # read infogain file. Format of each line is: word<TAB>infoGain
    # e.g., minute	0.9
    def read_info_gain(self, file, threshold):
        important_words = []
        with open(file) as f:
            for line in f.readlines():
                word, score = line.strip().split("\t")

                if float(score) > threshold:
                    important_words += [word]

        return important_words

    def read_tree(self, s):
        root, position = self.read_node(s, 0)
        return Tree(root)

    def read_node(self, s, position):
        has_children = False
        next_token = self.peek_next_token(s, position)

        if next_token == "[":  # the node we are going to read has children
            (token, position) = self.consume_next_token(s, position)
            has_children = True

        node_val, position = self.read_node_val(s, position)
        node = Node(node_val)

        if not has_children:
            return (node, position)

        while True:
            if self.peek_next_token(s, position) == "]":
                token, position = self.consume_next_token(s, position)
                break

            rel, position = self.read_relation(s, position)
            child, position = self.read_node(s, position)

            node.add_child((rel, child))

        return (node, position)

    def peek_next_token(self, s, position):
        if position == len(s):
            return None

        start = position
        tokens_a = [" ", "\t", "\n", "\r"]

        while start < len(s) and s[start] in tokens_a:
            start += 1

        if position >= len(s):
            return None

        end = start
        tokens_b = ["[", "]", ">", "/"]

        if s[start] in tokens_b:
            return s[start]

        while end < len(s) and s[end] not in tokens_a + tokens_b:
            end += 1

        return s[start:end].strip()

    def consume_next_token(self, s, position):
        if position == len(s):
            return None

        start = position
        tokens_a = [" ", "\t", "\n", "\r"]

        while start < len(s) and s[start] in tokens_a:
            start += 1

        if position >= len(s):
            return None

        end = start
        tokens_b = ["[", "]", ">", "/"]

        if s[start] in tokens_b:
            return (s[start], start + 1)

        while end < len(s) and s[end] not in tokens_a + tokens_b:
            end += 1

        return (s[start:end].strip(), end)

    def read_relation(self, s, position):
        start = position
        end = start

        while end < len(s) and s[end] != ">":
            end += 1

        if s[end] != ">":
            print("!! warning: > is missing from relation!")

        return (s[start:end].strip(), end + 1)

    def read_node_val(self, s, position):
        position1 = position
        word, position = self.consume_next_token(s, position)
        token, position = self.consume_next_token(s, position)

        if token != "/":
            print("!!! warning: wrong token separating word/pos. Token is " +
                  "%s at position %s in str %s with initial position %s" %
                  (token, position, s, position1))

        pos, position = self.consume_next_token(s, position)
        return ((word, pos), position)

    # read parse trees. parse trees for each sentence are separated by this
    # line: ---------, however we don't care about this yet. We just read all
    # parse trees. we can potentially store sentences related to parse trees
    # too, but the current input file doesn't contain sentences.
    def load_trees_from_file(self, file):
        # each new tree starts with a line starting with "[" (i.e., no
        # indentation)
        sentence_trees = []
        tree_str = ""

        with open(file) as f:
            for line in f.readlines():
                if line.strip() == "" or "---" in line.strip():
                    continue
                if line[0] == "[":
                    # new tree found. finish prev tree,
                    # add it to list and start a new one
                    if tree_str != "":
                        sentence_trees += [tree_str]

                    # the line we just read is the fisrt line of the new tree
                    tree_str = line.strip()

                else:
                    tree_str += " " + line.strip()

        if tree_str != "":
            sentence_trees += [tree_str]

        return sentence_trees

    def load_trees_from_file_with_sentences_starting_with_star(self, file):
        # each new section starts with a *<sentence>
        sentence_trees = []
        tree_str = ""
        sentence = ""
        reading_whitespace = False

        with open(file) as f:
            for line in f.readlines():
                if reading_whitespace:
                    if line != "" and (line.strip() == "" or
                                       line.strip()[0] != "["):
                        continue
                    else:
                        reading_whitespace = False
                elif line.strip() != "" and line.strip()[0] == "*":
                    # new sentence found. finish prev tree,
                    # add it to list and start a new one
                    if tree_str != "":
                        sentence_trees += [(sentence,
                                            self.read_tree(tree_str))]
                        tree_str = ""

                    sentence = line.strip()
                    # read remaining lines until we hit the beginning of
                    # a tree which is signaled by a line starting with "["
                    reading_whitespace = True
                    continue

                if line.strip() != "":
                    tree_str += line

        if tree_str != "":
            sentence_trees += [(sentence, self.read_tree(tree_str))]
        return sentence_trees

    def sort_dict(self, dictionary):  # dict is key->count
        l = []

        for key, value in dictionary.items():
            l += [(value, key)]

        return sorted(l, reverse=True)

    def add_suffix_to_all_non_star_nodes(self, node, i_list):
        i = i_list[0]
        i_list[0] = i + 1
        word, pos = node.node_val

        if word != "*":
            node.node_val = ("%s__%s" % (word, i), pos)

        for rel, child in node.children:
            self.add_suffix_to_all_non_star_nodes(child, i_list)

    def remove_suffixes(self, node):
        word, pos = node.node_val
        index = word.find("__")

        if index > 0:
            word = word[:index]
            node.node_val = (word, pos)
        for rel, child in node.children:
            self.remove_suffixes(child)

    def create_word_combinations_by_removing_one_word(self, words):
        return_list = []

        for word_to_remove in words:
            comb = []

            for word in words:
                comb += [word]

            comb.remove(word_to_remove)

            return_list += [comb]

        return return_list

    def create_sub_patterns(self, pattern_tree, patterns_so_far):
        # patterns = []
        important_word_nodes = {}

        # this will fill word_nodes with data
        pattern_tree.get_all_important_word_nodes(important_word_nodes)

        important = important_word_nodes.keys()

        if len(important) <= 2:
            # if patternTree has only 2 important words,
            # then we don't create subpatterns for it
            return

        combos = self.create_word_combinations_by_removing_one_word(important)

        for word_comb in combos:
            important = word_comb
            self.create_all_patterns_and_sub_patterns(pattern_tree, important,
                                                      patterns_so_far)

    def get_unique_patterns(self, patterns):
        # we create str representation of pattern subtrees
        # and add them to a set to remove duplicates
        pattern_str_patterns = {}

        for pattern in patterns:
            pattern_str = str(pattern)

            if pattern_str not in pattern_str_patterns:
                pattern_str_patterns[pattern_str] = pattern

        return pattern_str_patterns.values()

    def create_all_patterns_and_sub_patterns(self, node1, important_words,
                                             patterns_so_far):
        if len(node1.children) == 0:
            return

        node = node1.clone()
        # prunes the tree to keep only branches that contain an important word.
        # non-important words along the way are replaced by "*". For negation,
        # we don't care about the word; instead we only consider the relation
        # "neg", so wee keep branches containing a "neg" relation too.
        sub_tree = self.get_sub_tree(node, important_words)

        if sub_tree:
            if str(sub_tree) in patterns_so_far:
                return

            sub_tree2 = self.prune_empty_roots(sub_tree)

            self.make_indir_relations(sub_tree2)

            if len(sub_tree2.children) == 0:
                return

            if str(sub_tree2) in patterns_so_far:
                return

            patterns_so_far[str(sub_tree2)] = sub_tree2

            self.create_sub_patterns(sub_tree2, patterns_so_far)

    def extract_patterns(self, important_words, trees):
        pattern_str_count = {}
        patterns = []

        for tree in trees:
            # we clone the original tree, because we are going to change the
            # clone in place
            root_clone = self.read_tree(tree).root.clone()
            # initial prune of the parse tree to obtain only nodes that are
            # members of important_words
            sub_tree = self.get_sub_tree(root_clone, important_words)
            if not sub_tree or len(sub_tree.children) == 0:
                # no important word found, or there is only one node in subtree
                continue

            # now pattern_tree's words are replaced by word_suffix and hence
            # repeated words become unique. suffx is __i where i is a number
            # counting up from 1
            self.add_suffix_to_all_non_star_nodes(sub_tree, [0])

            important_word_nodes = {}

            # this will fill word_nodes with data. We want to extract all
            # important words (plus suffixes) that occur in this tree.
            sub_tree.get_all_important_word_nodes(important_word_nodes)

            # words_suffixes are important words plus suffixes.
            words_suffixes = important_word_nodes.keys()
            pattern_str_pattern = {}

            # this will fill allPatterns
            self.create_all_patterns_and_sub_patterns(sub_tree, words_suffixes,
                                                      pattern_str_pattern)

            for pattern_str in pattern_str_pattern:
                pattern_node = pattern_str_pattern[pattern_str]

                self.remove_suffixes(pattern_node)

                new_pattern_str = str(pattern_node)

                if new_pattern_str not in pattern_str_count:
                    pattern_str_count[new_pattern_str] = 0

                pattern_str_count[new_pattern_str] += 1

        # sort patterns by count and return
        for count, pattern in self.sort_dict(pattern_str_count):
            if pattern[0] == "(":
                patterns.append(pattern[1:-1].replace("  ", " "))
            else:
                patterns.append(pattern.replace("  ", " "))

        return patterns


# ========== main program===============

# --set main parameters and input files here-------------

#info_gain_threshold = 0.0025
#word_infogain_file = "/Users/lesani/Dropbox/work/Opinion-Mining/Data/word_infoGain/LongWaitTime_words.txt"  # "shortWait_word_infoGain.txt"
#tree_file = "/Users/lesani/Dropbox/work/Opinion-Mining/Data/sentences/LongWaitTime_trees.txt"
#pattern_extractor = PatternExtractor()
#important_words = pattern_extractor.read_info_gain(word_infogain_file,
#                                                   info_gain_threshold)
#
#trees = pattern_extractor.load_trees_from_file(tree_file)
#
#
## ---- for each parse tree, extract a pattern and all sub-patterns and add them
## to patternStr_count. Each extracted pattern is a node (with its children),
## but we store its string representation in patternStr_count.
#
#for pattern in pattern_extractor.extract_patterns(important_words, trees):
#    print(pattern)
