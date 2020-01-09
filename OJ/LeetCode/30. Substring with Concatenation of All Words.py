class Solution:
	def findSubstring(self, s, words):
		import collections
		if not words or not s:
			return []

		# initialize word frequency counter
		counter = collections.Counter(words)
		word_len = len(words[0])
		total_word_len = word_len * len(words)  # the len of the combination  

		result = []
		# iterate through all non-overlapping windows (e.g. modulus word_len)
		for i in range(word_len):
			# construct the initial substring for this window
			sub = s[i: i + total_word_len]

			# break down substring into individual words, construct counter from this
			sub_counter = collections.Counter(
				[sub[j: j + word_len] for j in range(0, len(sub), word_len)])

			# if substring counter matches words counter, add the index to result
			if counter == sub_counter:
				result.append(i)

			# slide the window over one word at a time
			for j in range(i + word_len, len(s) - total_word_len + 1, word_len):
				oldJ = j - word_len

				# remove instance of the old word from counter
				sub_counter[s[oldJ: j]] -= 1
				if sub_counter[s[oldJ: j]] == 0:
					# if reaches zero delete the key so counter equality works
					del sub_counter[s[oldJ: j]]

				# add instance of new word to the counter
				sub_counter[s[oldJ + total_word_len: j + total_word_len]] += 1

				# if substring counter matches words counter, add the index to result
				if counter == sub_counter:
					result.append(j)

		return result
