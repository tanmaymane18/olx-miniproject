import requests
from bs4 import BeautifulSoup


def removeStr(x):
    s = ""
    for c in x:
        if c.isnum():
            s += c

    return s


def getAttrs(url):
    headers = {
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'}
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')


    attrs = soup.find_all(class_="_2vNpt")
    brand = attrs[0].text
    model = attrs[1].text
    year = attrs[2].text
    running = attrs[3].text.replace(",", '').split()[0]
    price = soup.find(class_="_2xKfz").text.replace(",", '').split()[1]
    print(brand, model, year, running, price)

    return brand, model, year, running, price
