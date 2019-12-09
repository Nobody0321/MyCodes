class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        numbers = sorted(nums)
        premutations = []
        def generateP(numbers, result):
            base = numbers[0]
            for each in numbers[1:]:
                result.append()