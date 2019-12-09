# 堆排序其实没有真正定义二叉树的结构，但是在数组中使用二叉树的思想进行遍历和排序
# 对所有节点倒序排序，这样保证最大值/最小值一定是从最底层层层上升到最高层
class Heap_Sort:
    def __init__(self, array, increase=True):
        # array= [99, 5, 36, 7, 22, 17, 46, 12, 2, 19, 25, 28, 1, 92]
        self.arr = array
        self.increase = increase

    def heap_sort(self):   
        l = len(self.arr) - 1
        i = l >> 1  # 从最后一个非叶子节点开始（堆是二叉树，叶子最多有(N + 1) / 2）
        # 1. 构建 最大/最小 堆
        while i >= 0:
            self.adjust_heap(i, l)
            i -= 1
        
        # 2. 取出最大/最小元素，剩下元素重新排序
        while l:
            # 最值 现在一定在堆顶，可以放到数组尾部，然后对剩下的n-1个元素排序
            self.arr[l], self.arr[0] = self.arr[0], self.arr[l]
            l -= 1
            self.adjust_heap(0, l)

        return self.arr

    def adjust_heap(self, i, l):
        # 从数组中构建堆(升序构建大顶堆，降序构建小顶堆)
        child_idx = i * 2 + 1  # 该节点的左儿子
        while child_idx <= l:
            if self.increase:
                if child_idx + 1 <= l and self.arr[child_idx] < self.arr[child_idx + 1]:
                    # 如果有右儿子结点且更大，切换到右儿子结点
                    child_idx += 1
                if self.arr[child_idx] > self.arr[i]:
                    # 子节点中较大的比父节点大，那就交换
                    self.arr[i], self.arr[child_idx] = self.arr[child_idx], self.arr[i]
                    i = child_idx
                    child_idx = child_idx * 2 + 1
                else:
                    # 后边也不用比了
                    break
            else:
                if child_idx + 1 <= l and self.arr[child_idx] > self.arr[child_idx + 1]:
                    # 如果有右儿子结点且更大，切换到右儿子结点
                    child_idx += 1
                if self.arr[child_idx] < self.arr[i]:
                    # 子节点中较大的比父节点大，那就交换
                    self.arr[i], self.arr[child_idx] = self.arr[child_idx], self.arr[i]
                    i = child_idx
                    child_idx = child_idx * 2 + 1
                else:
                    # 后边也不用比了
                    break

    
if __name__ == "__main__":
    
    import random
    nums = [random.randint(1,100) for _ in range(20)]
    print(nums)
    S = Heap_Sort(nums)
    print(S.heap_sort())