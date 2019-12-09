if __name__ == "__main__":
    n, p, q = list(map(int, input().strip().split()))
    total = 0
    ans = 0

    def combine(n_, c):
        if n_ == c:
            return 1
        if c == 0:
            return 1
        n1 = 1
        for i in range(n_-c, n+1):
            n1 *= i
        n2 = 1
        for i in range(1, c+1):
            n2 *= i
        return n1/n2

    for j in range(p, n-q+1):
        num_of_iter = combine(n, j)
        total += num_of_iter
        ans += num_of_iter * j
    k = (ans % total) / total
    ret = int(1000000007 * k)
    if k != 0:
        ret += 1000000007 - int(ret / k)
    print(ret)