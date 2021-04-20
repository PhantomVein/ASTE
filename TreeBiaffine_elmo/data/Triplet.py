class Triplet:
    polar_id2tag = ['NEU', 'POS', 'NEG']
    polar_tag2id = {'NEU': 0, 'POS': 1, 'NEG': 2}

    def __init__(self, target, polar, term, dependency):
        self.target = target
        self.polar = polar
        self.term = term
        self.dependency = dependency

    def __str__(self):
        target = ' '.join([self.dependency[id].org_form for id in self.target])
        term = ' '.join([self.dependency[id].org_form for id in self.term])
        polar = self.polar_id2tag[self.polar]
        triple_str = '({}, {}, {})'.format(target, polar, term)
        return triple_str

    def str_id(self):
        triple_str = '({}, {}, {})'.format(self.target, self.term, self.polar)
        return triple_str

    def __eq__(self, other):
        # `__eq__` is an instance method, which also accepts one other object as an argument.
        if self.str_id() == other.str_id():
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.str_id())
