test_ocr_result = ['', '', 'Suggested USe: Adults tale 1 sottgel two to three times daily with meals,', 'or as recommended by your healthcare provider. CAUTION. As with dietan sunnlement sh anv VOU', 'CAUTION: As with any dietary supplement, you should consult your healthcare provider before use. especially if pregnant, nursing, have a', 'cspetlaly I llave d medical condition, taking medications, or have known adverse reactions', 'oralergies', 'Supplement Facts', 'supplement olemen Serving Size: 1 Softgel', 'Serving Size: 1 Softgel Servings Per Container: 240', 'Amount Per % Daily', 'Serving Value"', 'Calories 1', 'Total Fat 1g 1%', 'Polyunsaturated Fat', 'Certified Organic Flaxseed Oil 1200 mg', '(Linum usitatissimum) Alpha-Linolenic Acid (Omega-3) 720 mg', 'AIpha-Linolenic Acid (omega-3) 720 mg Linoleic Acid (Omega-6) 120 mg', 'Oleic Acid (Omega-9 140 mg', '"Percent Daily Values are based on a 00-calorie diet.', 'tt Daily Value not established.', 'Other Ingredients: Softgel capsule (halal gelatin, glycerin,', 'purified water). This Product Does Not C', 'This Product Does Not Contain: Wheat, gluten, dairy, soy, eggs, tree nuts, peanuts, fish, or shellfish.', 'KEEP OUT OF THE REACH OF CHILDREN. DO NOT USE IF SAFETY SEAL IS', 'DAMAGED OR MISSING. STORE IN A COOL, DRY PLACE. t Supportive but not conclusive research shows that consumption of EPA', 'supporive DUT not conclusive research shows that consumption EPA and DHA omega- -3 fatty acids may reduce the risk of coronary heart disease.', 'NatureWise Organic Flaxseed Oil should always be taken in conjunction', 'with healthy diet and regular exercise program.', 'These statements have not been evaluated by the Foodt and Drug', 'Administration. This product is not intended to diagnose, treat, cure, or prevent any disease.', 'any Ustast. 2017 NatureWise. All Rights Reserve', 'Wise. All Rights Reserved.']
s = ' '.join(test_ocr_result)
print(s)
import spacy
nlp = spacy.load('en_core_web_sm')

token = nlp(s)
print(token)
# import nltk
# facts = []

# # for s in(test_ocr_result):
# #     if s == '':
# #         continue
    
# # s = ' '.join(test_ocr_result)
# word_token= nltk.word_tokenize(s) # remove stopwords
# print(word_token)

# word_tag = nltk.pos_tag(word_token)
# print(word_tag)

# ners = nltk.ne_chunk(word_tag, binary=False)  # 实体识别
# print(ners)

# for i in range(len(word_tag)):
#     word, tag = word_tag[i]
#     if tag == 'CD':
#         if i != len(word_tag)-1 and word_tag[i+1][1] == 'NN':
#             word += word_tag[i+1][0]
#         facts.append(word)

# print(facts)