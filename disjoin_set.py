import logging


logging.basicConfig(format="%(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DisjointException(Exception):
    pass


class DisjointUnionNode:
    def __init__(self, value: int, mark, manager):
        self.root = None
        self.mark = mark
        self.value = value
        self.manager = manager
        self.rank = 0

    def get_root(self):
        return self.root

    def get_mark(self):
        return self.mark

    def set_root(self, root):
        self.root = root

    def increment_rank(self):
        self.rank += 1

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def get_rank(self):
        return self.rank

    def find(self):
        r_root = self
        while r_root.get_root():
            r_root = r_root.get_root()
        if id(r_root) != id(self):
            self.set_root(r_root)
        return r_root

    def union(self, s):
        if isinstance(s, DisjointUnionNode) and id(self.manager) == id(s.manager):
            r_root = self.find()
            s_root = s.find()

            if id(r_root) == id(s_root):
                return self
            else:
                if r_root.get_rank() == s_root.get_rank():
                    s_root.set_root(r_root)
                    self.increment_rank()
                    r_root.set_value(max(s_root.get_value(), r_root.get_value()))
                    return self
                elif r_root.get_rank() > s_root.get_rank():
                    r_root.set_root(s_root)
                    s_root.set_value(max(s_root.get_value(), r_root.get_value()))
                    return s
                else:
                    s_root.set_root(r_root)
                    r_root.set_value(max(s_root.get_value(), r_root.get_value()))

        else:
            raise DisjointException("unknown params")


class DisjointUnionManager:
    def __init__(self):
        self.set = {}
        self.count = 1

    def create_new_union(self, value):
        self.set[self.count] = DisjointUnionNode(value, self.count, self)
        self.count += 1
        return self.count - 1

    def get_union_by_number(self, x: int):
        if x in self.set:
            return self.set[x]
        else:
            return None

    def get_count(self) -> int:
        return self.count - 1

if __name__ == "__main__":
    pass