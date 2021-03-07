import torch

class Node:
    def __init__(self, data, _pre=None, _next=None):
        self.data = data # in format [loss, feature]
        # self.data = data[0]
        # self.loss = data[1]
        # self.index = data[2]
        self._pre = _pre
        self._next = _next
    def __str__(self):
        return str(self.data[2])

class DoublyLink:
    def __init__(self):
        self.tail = None
        self.head = None
        self.size = 0

    # at the end
    def append(self, new_node):
        tmp_node = self.tail
        tmp_node._pre = new_node
        new_node._next = tmp_node
        new_node._pre = None
        self.tail = new_node
        return new_node

    # at the head
    def add_first(self, new_node):
        tmp_node = self.head
        tmp_node._next = new_node
        new_node._pre = tmp_node
        new_node._next = None
        self.head = new_node
        return new_node

    def insert_before(self, node, new_node):
        node._next._pre = new_node
        new_node._next = node._next
        new_node._pre = node
        node._next = new_node
        return new_node

    def insert_after(self, node, new_node):
        if node._pre is None:
            return self.append(new_node)
        else:
            return  self.insert_before(node._pre, new_node)
            # node._next = new_node
            # new_node._next = None
            # new_node._pre = node
            # self.head = new_node
            # return new_node

    def insert(self, data):
        if isinstance(data, Node):
            tmp_node = data
        else:
            tmp_node = Node(data)
        if self.size == 0:
            self.tail = tmp_node
            self.head = self.tail
        else:
            # pre_node = self.head
            tmp_node = self.add_first(tmp_node)
            # while pre_node.data[0] > tmp_node.data[0] and pre_node._pre != None:
            #     pre_node = pre_node._pre
            # #insert before
            # # print(pre_node._pre, pre_node._next)
            # if pre_node._pre is None and pre_node.data[0] >= tmp_node.data[0]:
            #     tmp_node = self.append(tmp_node)
            # elif pre_node._next is None and pre_node.data[0] < tmp_node.data[0]:
            #     tmp_node = self.add_first(tmp_node)
            # elif pre_node._next is None and pre_node.data[0] >= tmp_node.data[0]:
            #     tmp_node = self.insert_after(pre_node, tmp_node)
            # else:
            #     tmp_node = self.insert_before(pre_node, tmp_node)
        self.size += 1
        return tmp_node

    def remove(self, node):
        if node == self.head:
            self.head._pre._next = None
            self.head = self.head._pre
        elif node == self.tail:
            self.tail._next._pre = None
            self.tail = self.tail._next
        else:
            node._next._pre = node._pre
            node._pre._next = node._next
        self.size -= 1

    def __str__(self):
        str_text = ""
        cur_node = self.head
        count = 0
        while cur_node != None:
            str_text += str(cur_node.data[2]) + " "
            cur_node = cur_node._pre
            count += 1
            if count > 20:
                break
        return str_text


class LRUCache:
    def __init__(self, size):
        self.size = size
        self.hash_map = dict()
        self.link = DoublyLink()
        self.LRU_init(size)

    def LRU_init(self, size):
        for i in range(size):
            self.set(i, [1e-8, torch.rand(128), i])

    def set(self, key, value):
        if self.size == self.link.size:
            self.link.remove(self.link.tail)
        if key in self.hash_map:
            self.link.remove(self.hash_map.get(key))
        tmp_node = self.link.insert(value)
        self.hash_map.__setitem__(key, tmp_node)

    def get(self, key):
        tmp_node = self.hash_map.get(key)
        self.link.remove(tmp_node)
        self.link.insert(tmp_node)
        return tmp_node.data

    def get_queue(self, num, keys):
        queue = torch.rand(num, 128).cuda()
        num_queue = 0
        cur_node = self.link.head
        while num_queue < num:
            if cur_node.data[2] not in keys:
                queue[num_queue] = cur_node.data[1]
                num_queue += 1
            cur_node = cur_node._pre
        # while num_queue < num:
        #     # if cur_node.data[2] not in keys:
        #     queue[num_queue] = cur_node.data
        #     num_queue += 1
        #     cur_node = cur_node._next
        return queue

    def update_queue(self, queue):
        num = queue.size(0)
        # print(num)
        cur_node = self.link.head
        for i in range(num):
            queue[i] = cur_node.data[1]
            cur_node = cur_node._pre
        return queue

    def batch_set(self, keys, values, losses):
        # print(self.link)
        num = len(values)
        for i in range(num):
            self.set(keys[i], [losses[i].item(), values[i], keys[i]]) # add loss,data,key

# r = LRUCache(3)
# r.set("1", ["1","1"])
# r.set("2",  ["2","2"])
# r.set("3",  ["3","3"])
# print(r.link.size)
# r.get("1")
# print(r.link)
# r.set("4",  ["4","4"])
# print(r.link)

