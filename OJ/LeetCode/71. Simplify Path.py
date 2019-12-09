class Solution:
    def simplifyPath(self, path):
    # 1. 最后一个/要去掉  2. 最高目录是/  3. 多重///化简成一层 如a///b->a/b
    # 思路：1.分别确定.. 与.  2. 使用两个栈  3.类似网易字符串
        l1, l2 = [], []
        tmp = ''
        for ch in path:
            if ch == '/':
                # 刚开始 或者 前面的文件夹名称部分结束了
                if tmp != '':
                    l2.append(tmp)
                    tmp = ''
                
                l1.append(ch)

            if ch == '.':
                if l1 and l1[-1] == '.':
                    # 是'..' 去掉最后的’//..‘
                    l1 = l1[:-2]
                    if l1 == []:
                        l1 = ['/']
                    if l2:    
                        l2.pop() # 有'..'去掉最后一个文件夹
                if l1 and l1[-1] == '/':
                    l1.pop()
                l1.append(ch)
            
            if ch not in ('/','.'):
                # 如果是名称部分，那就开始缓存名称
                tmp += ch

        print(l1, l2)
        
        if l2 == []:
            return '/'
        return '/'+'/'.join(l2)


    def simplifyPath_2(self, path):
        # 解法2 ，用python
        p, stack = [p for p in path.split('/') if p != '' and p != '.'], []
        for each in p:
            if each == '..' and stack: stack.pop()
            elif each != '..': stack.append(each)
        print(stack)
        return "/" + "/".join(stack)

if __name__ == "__main__":
    print(Solution().simplifyPath_2("/a/./b/../../c/"))          