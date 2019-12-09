class Solution:
    def restoreIpAddresses(self, s):
        self.result = []        
        self.doRestore(s, [])
        return self.result

    def doRestore(self, s, path):
        if s == '' or len(path) == 4:
            # 遇到了一个坑, python里面''不是None
            if s == '' and len(path) == 4:
                # 如果s为空且已经有了四个有效分组,可以停止回溯
                self.result.append('.'.join(path))
            return
        else:
            # 如果一个数是0，后面不能跟其他数字,不然的话可以再带最多2个数
            to = 1 if s[0] == '0' else 3
            
            for i in range(1, min(to, len(s))+1):
                # 分情况遍历1-3个数字
                part = s[:i]
                if int(part) <= 255:
                    self.doRestore(s[i:], path + [part])

    
if __name__ == "__main__":
    print(Solution().restoreIpAddresses("25525511135"))