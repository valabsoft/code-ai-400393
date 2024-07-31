import requests
url = 'https://yandex.ru/images/search?from=tabbar&text=самолеты'
 


response = requests.get(url)
print(response.content)
#file_Path = '../body.html'
 
#if response.status_code == 200:
#    with open(file_Path, 'wb') as file:
#        file.write(response.content)
#    print('File downloaded successfully')
#else:
#    print('Failed to download file')
