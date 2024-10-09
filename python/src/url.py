import requests
 
#if response.status_code == 200:
import requests
url = 'https://yandex.ru/images/search?from=tabbar&text=самолеты'
 


response = requests.get(url)
print(response.content)
#file_Path = '../body.html'
 
#if response.status_code == 200:
