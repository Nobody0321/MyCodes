import requests
login_info = {
            "phone": "18502208636",
            "password": "13072002394cxt",
            "shop_id": None
        }
login_url = 'https://snailpet.com/v2/Passport/login'

def test():
    session = requests.Session()
    r = session.post(login_url,data=login_info)
    return r.cookies


class Snail_Spider:
    
    def __init__(self):
        self.login_info = {
            "phone": "18502208636",
            "password": "13072002394cxt",
            "shop_id": None
        }
        self.login_url = 'https://snailpet.com/v2/Passport/login'
        self.target_url = 'https://snailpet.com/v2/Members/getList?page=1&keywords=&birthday=&petSpecies=&petAge=&balance=&lastConsumption=&sex=&levelId=-1&orderBy=updated&is_mini_login=0&shopId=7717&shop_id=7717'

    def get_token(self):
        session = requests.Session()
        r = session.post(login_url,data=login_info)

        cookieStr = ''

        for item in r.cookies:
            cookieStr = cookieStr + item.name + '=' + item.value + ';'

        return cookieStr

    def get_member(self):
        headers  ={
            # 'cookie' : 'version=2017.08.04; snailServerTimeC=-1; snailLoginPhone=18502208636; snailToken=c6454c2affac12f48c26850d13b8842a; snailNowUserId=11738; snailShopId=7717; snailPrint=1%2C2%2C3%2C7%2C8%2C9%2C10%2C11; snailShopInfo=%7B%22create_user_id%22%3A11738%2C%22name%22%3A%22%E6%B4%BE%E5%A4%9A%E6%A0%BC%E5%AE%A0%E7%89%A9%E5%A4%A9%E6%B4%A5%E4%B8%87%E7%A7%91%E6%B0%91%E5%92%8C%E5%B7%B7%E5%BA%97%22%2C%22shop_phone%22%3A%2218502208636%22%2C%22weixin%22%3A%2213512863603%22%2C%22logo%22%3A%22https%3A%2F%2Ffile.snailpet.cn%2Fb8%2Fbe%2Fb8be1bed5ffb160af4faec11beead610.jpg%22%2C%22address%22%3A%22%E5%A4%A9%E6%B4%A5%E5%B8%82%E4%B8%9C%E4%B8%BD%E5%8C%BA%E4%B8%87%E7%A7%91%E6%B0%91%E5%B7%B7%E8%8A%B1%E5%9B%AD%E5%BA%95%E5%95%863-106%22%2C%22desc%22%3A%22%22%2C%22is_authentication%22%3A1%2C%22signs_image%22%3A%22https%3A%2F%2Ffile.snailpet.cn%2F7e%2F0d%2F7e0dc2dd87132c2f9f8f344f25744ee6.jpg%22%2C%22license_image%22%3A%22%22%2C%22print%22%3A%221%2C2%2C3%2C7%2C8%2C9%2C10%2C11%22%2C%22qr_code%22%3A%22https%3A%2F%2Fu.wechat.com%2FME6zLF7a-y73VccUsx03VNA%22%2C%22print_slogan%22%3A%22%22%2C%22shop_type%22%3A0%2C%22invite_code%22%3A%22%22%2C%22invite_url%22%3A%22%22%2C%22exp_status%22%3A0%2C%22number%22%3A%2226089%22%2C%22shop_plus%22%3Anull%2C%22create_user_info%22%3Anull%2C%22user_shop_id%22%3A13984%2C%22user_id%22%3A11738%2C%22shop_id%22%3A7717%2C%22type%22%3A1%2C%22created%22%3A1533632123%2C%22is_default%22%3A1%2C%22powers%22%3A%22%22%2C%22opt_user_id%22%3A11738%2C%22able_order%22%3A1%2C%22title%22%3A%22%22%2C%22home_set%22%3A%22%22%2C%22product_percentage%22%3A0%2C%22server_percentage%22%3A0%2C%22basic_salary%22%3A0%2C%22tasks%22%3Anull%7D; snailPowers=1; isHighUser=1; snailShopLevel=0; snailReadRen=%7B%22id%22%3A3210%2C%22shop_id%22%3A7717%2C%22apply_user_id%22%3A11738%2C%22name%22%3A%22%E6%B4%BE%E5%A4%9A%E6%A0%BC%E5%AE%A0%E7%89%A9%E5%A4%A9%E6%B4%A5%E4%B8%87%E7%A7%91%E6%B0%91%E5%92%8C%E5%B7%B7%E5%BA%97%22%2C%22phone%22%3A%2218502208636%22%2C%22address%22%3A%22%E5%A4%A9%E6%B4%A5%E5%B8%82%E4%B8%9C%E4%B8%BD%E5%8C%BA%E4%B8%87%E7%A7%91%E6%B0%91%E5%B7%B7%E8%8A%B1%E5%9B%AD%E5%BA%95%E5%95%863-106%22%2C%22weixin%22%3A%2213512863603%22%2C%22signs_image%22%3A%22https%3A%2F%2Ffile.snailpet.cn%2F7e%2F0d%2F7e0dc2dd87132c2f9f8f344f25744ee6.jpg%22%2C%22license_image%22%3A%22%22%2C%22mark%22%3A%22%22%2C%22apply_time%22%3A1533632952%2C%22reply_time%22%3A1533695963%2C%22status%22%3A2%2C%22is_read%22%3A1%2C%22reason%22%3A%22%22%7D',
            'cookie': 'version=2017.08.04; snailServerTimeC=-1; snailLoginPhone=18502208636; snailToken=c6454c2affac12f48c26850d13b8842a',
            'token': 'c6454c2affac12f48c26850d13b8842a',
            'signature': 'e68f6055483c4b8231dd15eae634fbed',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36',
            'referer': 'https://snailpet.com/member',
            'request-time': '1546159728'

        }
        session = requests.Session()
        r = session.get(self.target_url, headers = headers)
        return r.text

if __name__ == '__main__':
    s = Snail_Spider()
    print(s.get_member())