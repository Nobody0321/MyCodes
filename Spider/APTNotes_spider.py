save_folder = 'd:/APTNotes/'
host_url = 'http://www.cnnvd.org.cn'
target_url = 'http://www.cnnvd.org.cn/web/vulreport/queryListByType.tag'
test_url = host_url + '/web/cnnvdform/ldbgxz.tag?filePath=20190121170612294'
page = '?pageno=2'

import requests
from pyquery import PyQuery as pq
import time

def downlaod_from_single_website(url):
    
    res = requests.get(url)
    pqObj = pq(res.content)
    titles = pqObj('a[class="a_title"]').items()
    for title in titles:
        function = title.attr('onclick')
        filename = title.text()
        pdf_url = 'http://www.cnnvd.org.cn/web/cnnvdform/ldbgxz.tag?filePath='+ function.split('\'')[1]
        
        r = requests.get(pdf_url)

        with open(save_folder+ filename + '.pdf', "wb") as f:
            f.write(r.content)
            print('download complete:', filename, '\nform:', pdf_url)

if __name__ == '__main__':
    urls = [target_url]
    for i in range(2,51):
        urls.append(target_url + '?pageno=' +str(i))

    for url in urls:
        downlaod_from_single_website(url)
        time.sleep(10) 