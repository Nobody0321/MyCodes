class Solution:
    def combine(self, n, k):
        self.results = []
        self.conbine([], list(range(1, n+1)), k)
        return self.results
    
    def conbine(self, path, ran, k):
        print(path, ran, k)
        if k == 0:
            self.results.append(path)
            return
        elif ran == []:
            return
        else:
            # with the first
            self.conbine(path + [ran[0]], ran[1:], k-1)
            # without the first
            self.conbine(path, ran[1:], k)

if __name__ == "__main__":
    print(Solution().combine(n=3,k=3))