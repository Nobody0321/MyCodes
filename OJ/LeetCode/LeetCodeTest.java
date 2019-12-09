import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

public class LeetCodeTest {
    public static void main(String[] args) {
        Solution s = new Solution();
        String input = "{}{}{}";
        Boolean result = s.isValid(input);
        System.out.println(result);
    }
}

class Solution {
    public boolean isValid(String s) {
        HashMap<Character,Integer> left = new HashMap<Character, Integer>{{
            left.put('(', 0);
            left.put('{', 1);
            left.put('[', 2);
        }};
        List<Character> right = new ArrayList<Character>{{
            right.add('(');
            right.add('{');
            right.add('[');
        }};
        List<Character> stack = new ArrayList<Character>();
        for (char x : s) {
            if(left.containsKey(x)){
                stack.add(x);
            }
            else if (right.contains(x) && stack.size()!=0){
                if (right.get(left.get(stack.peek())) != x) {
                    return false;
                }
            }
            else{
                return False;
            }
        }
        if (stack.size() != 0){
            return false;
        }
        return true;
    }
    public List<String> letterCombinations(String digits) {
        LinkedList<String> ans = new LinkedList<String>(); //先进先出（为什么用先进先出？）
        if (digits.isEmpty()) return ans;
        String[] mapping = new String[]{"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        ans.add("");
        for (int i = 0; i < digits.length(); i++) {
            int x = Character.getNumericValue(digits.charAt(i));
            while (ans.peek().length() == i) { // 有几个数字，最后结果每个字串都会变成多长
                String t = ans.remove();//取出ans最先进去的元素
                for (char s : mapping[x].toCharArray())
                    ans.add(t + s);
            }
        }
        return ans;
    }
    public class ListNode {
          int val;
         ListNode next;
          ListNode(int x) { val = x; }
    }

    public ListNode removeNthFromEnd_1(ListNode head, int n) {
        List<Integer> result= new ArrayList<Integer>();
        ListNode ret = new ListNode(0);
        ListNode retHead = ret;
        while(head!=null){
            result.add(head.val);
            head = head.next;
        }
        int length = result.size();
        for(int i=0;i<length;i++){
            if(i == length-n) continue;
            ret.next = new ListNode(0);
            ret = ret.next;
            ret.val = result.get(i);
        }
        return retHead.next;
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {
        //fast 先走n步，然后slow开始走，那么当fast走完的时候，slow恰好走到倒数第n
        ListNode start = new ListNode(0);
        ListNode slow = start, fast = start;//java的机制，slow、fast、start初始化指向同一个对象，称为a
        start.next = head;//这个a后面接上原始的listnode

        for(int i=0; i<=n; i++)   {
            fast = fast.next;//fast类似指针，一步步迭代，指向链表下一个节点
        }
        while(fast != null) {
            slow = slow.next;
            fast = fast.next;
        }
        slow.next = slow.next.next;
        return start.next;
    }
}