# T = int(input())
IN = [['7','3','3','4','4','5','6'],['2','3','4','6','A'],['A','2','3','4','5'],['3','4','5','6','7','8']]
T = 4
result = [0]*T
for _ in range(T):
    # N = int(input())
    # cards = input().split(' ')
    # cards = '7 3 3 4 4 5 6'.split(' ')
    cards = IN[_]
    # print(cards)
    d = {'A':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'J':11,'Q':12,'K':13}
    count = {1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0}
    for card in cards:
        count[d[card]] += 1
    for c in list(count.keys())[:-4]:
        t = 1
        for a in range(5):
            t *= count[c+a]
        if t:
            result[_] += t
            for a in range(5,13):
                try:
                    if count[c + a]:
                        t = 1
                        for i in range(a+1):
                            t *= count[c+i]
                        result[_] += t
                except:
                    break
for i in range(len(result)):
    print(result[i],end='')
    if i != len(result)-1:
        print()