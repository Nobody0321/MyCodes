class Solution {
    public boolean isValid(String s) {
        HashMap<Character,Integer> left = new HashMap<Character, Integer>();

        left.put('(', 0);
        left.put('{', 1);
        left.put('[', 2);

        List<Character> right = new ArrayList<Character>();

        right.add(')');
        right.add('}');
        right.add(']');

        Stack<Character> stack = new Stack<Character>();

        for (int i = 0; i < s.length(); i++) {

            char x = s.charAt(i);

            if (left.containsKey(x)) {
                stack.push(x);
            }

            else if (right.contains(x) && stack.size() != 0) {
                if (right.get(left.get(stack.pop())) != x) {
                    return false;
                }
            }

            else {
                return false;
            }
        }
        if (stack.size() != 0){
            return false;
        }
        return true;
    }
}