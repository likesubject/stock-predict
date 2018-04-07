# -- coding: utf-8 --
import urllib2
import execjs

class Stock(object):
    def __init__(self, url):
        self.url = url

    def _request_data(self, url):
        request = urllib2.Request(url)
        response = urllib2.urlopen(request)
        data = response.read()
        return data

    def get_all_list(self):
        info = []
        data = self._request_data(self.url)
        ctx = execjs.compile(data)
        stock_items = ctx.call("get_data")
        for stock in stock_items:
            info.append(stock['val'])
        return info

#test
stock = Stock('http://www.sse.com.cn/js/common/ssesuggestdata.js')
data = stock.get_all_list()
pass
