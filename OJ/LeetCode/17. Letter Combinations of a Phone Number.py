class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        digit_map = {
            '2': ["a","b","c"],
            '3': ["d", "e", "f"],
            '4': ["g","h","i"],
            '5': ["j","k","l"],
            '6': ["m", "n", "o"],
            '7': ["p","q","r", "s"],
            '8': ["t","u","v"],
            '9': ["w", "x", "y", "z"]
        }
        
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