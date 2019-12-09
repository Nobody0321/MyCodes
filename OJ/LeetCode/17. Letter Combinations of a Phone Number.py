class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        digit_map = {}
        digit_map['2'] = ["a","b","c"]
        digit_map['3'] = ["d", "e", "f"]
        digit_map['4'] = ["g","h","i"]
        digit_map['5'] = ["j","k","l"]
        digit_map['6'] = ["m", "n", "o"]
        digit_map['7'] = ["p","q","r", "s"]
        digit_map['8'] = ["t","u","v"]
        digit_map['9'] = ["w", "x", "y", "z"]
        
        result = []
        for digit in digits:
            result = self.get_combos(result, digit_map[digit])
            
        return result
        
    def get_combos(self, str1, str2):
        result = []
        
        if len(str1) <= 0:
            return str2
        elif len(str2) <= 0:
            return str1
        
        for letter1 in str1:
            for letter2 in str2:
                result.append(letter1 + letter2)
                
        return result

print(Solution().letterCombinations('23'))