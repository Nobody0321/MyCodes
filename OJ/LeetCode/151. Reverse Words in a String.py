class Solution:
    # def reverseWords(self, s: str) -> str:
    #     return " ".join(s.strip().split()[::-1])

    def reverseWords(self, s: str) -> str:
        words = []
        temp_word = ""
        word_flag = False
        for i in range(len(s)):
            if not word_flag and s[i] != " ":
                word_flag = True
                temp_word += s[i]

            elif word_flag and s[i] == " ":
                word_flag = False
                words.append(temp_word)
                temp_word = ""

            elif word_flag:
                temp_word += s[i]

        if word_flag and temp_word:
            words.append(temp_word)
            
        reversed_words = []
        for i in range(len(words) - 1, -1, -1):
            reversed_words.append(words[i])
            
        ret = ""
        for each in reversed_words:
            ret += each
            ret += " "
        return ret[:-1]


if __name__ == "__main__":
    s = "the sky is blue"
    sol = Solution()
    print(sol.reverseWords(s))
