# labelledFile = r"D:\数据集\CCKS2017\CCKS2017_dataset\case_of_illness\data\trainingset 1-100\病史特点\病史特点-1.txt"
# rawFile = r"D:\数据集\CCKS2017\CCKS2017_dataset\case_of_illness\data\trainingset 1-100\病史特点\病史特点-1.txtoriginal.txt"
# targetFile = r"D:\数据集\CCKS2017\CCKS2017_dataset\case_of_illness\data\trainingset 1-100\病史特点\病史特点-1.txtlabelled.txt"
import string
import os
punc = string.punctuation + u"！？＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
# print(punc)

def target(rawFile, labelledFile, targetFile):
    line = list(open(rawFile, encoding='utf-8').readline().strip())
    for i in range(len(line)):
        if line[i] in punc:
            line[i] = ''
    # print(line)
    labels = open(labelledFile, encoding='utf-8').readlines()
    labels = [line.strip().split('\t') for line in labels]

    for label in labels:
        entity, start, end, target = label
        for idx in range(int(start), int(end)+1):
            if line[idx]:
                line[idx] += '\t'+target

    with open(targetFile, 'w') as f:
        for chr in line:
            if chr in punc or len(chr) == 0:
                continue
            if len(chr) == 1:
                chr += '\t'+'O'
            f.write(chr+'\n')


def labeldata(dataDir):

    import os
    filenames = os.listdir(dataDir)
    rawfilenames = [os.path.join(dataDir, filename) for filename in filenames if 'original' in filename]
    labelledFile = [os.path.join(dataDir, filename) for filename in filenames if 'original' not in filename]

    for rawfilename in rawfilenames:
        target(rawfilename,  rawfilename.split('.')[0] + '.' + rawfilename.split('.')[-1], rawfilename.split('.')[0] + '.txtlabelled.'+ rawfilename.split('.')[-1])


def mergeData(destFile):
    # if not os.path.exists(destFile):
    #     os.makedirs(destFile)
    filenames = os.listdir(dataDir)
    labelledFiles = [os.path.join(dataDir,filename )for filename in filenames if 'labelled' in filename]
    for labelledFile in labelledFiles:
        print(labelledFile)
        lines = open(labelledFile, encoding='gbk').readlines()
        with open(destFile, 'a+', encoding='gbk') as f:
            f.writelines(lines)


if __name__=="__main__":
    dataDir = r"./data/病史特点"
    corpusFile = r"./data/corpus.txt"

    labeldata(dataDir)
    mergeData(corpusFile)