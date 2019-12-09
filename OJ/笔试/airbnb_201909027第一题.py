def costsOfNodes(lines):
    # Write your code here
    trie = {}
    for line in lines:
        for i, k in enumerate(line.split(',')):
            if i == 0:
                if k not in trie:
                    trie[k] = []
                key = k
            else:
                if k in trie:
                    trie[k].append(key)
                else:
                    trie[k] = [key]
    def dfs(node, trie):
        if node == None:
            return
        if trie[node]:
            res.extend(trie[node])
            for each in trie[node]:
                dfs(each, trie)
    ret = []    
    for k in trie:
        res = []
        dfs(k, trie)
        trie[k] = list(set(res))
        ret.append(k +"," +str(len(trie[k]) + 1))

    ret = sorted(ret, key=lambda x: x[0])
    return ret
line = ["A,E,N,S","S,H,N","E,N","H","N"]
print(costsOfNodes(line))