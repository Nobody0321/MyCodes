class Solution:
    def FindKthToTail(self, head, k):
        if head == None or k < 1:
            return None
        tmp = head
        while k > 1:
            if tmp.next:
                tmp = tmp.next
                k -= 1
            else :
                return 
                
        while tmp.next:
            tmp = tmp.next
            head = head.next
        return head