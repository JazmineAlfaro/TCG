
from bs4 import BeautifulSoup
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium import webdriver
import time
import datetime
import pandas as pd
import sys, errno 
from requests.exceptions import HTTPError


def description_scraper(url):
    driver = webdriver.Firefox(executable_path=r'C:/Users/jazmi/Documents/geckodriver.exe')
    driver.get(url)
    #print(url)
    soup = BeautifulSoup(driver.page_source,'html.parser')
    driver.close()
    #print(soup)
    res = soup.find_all('div',class_="story-contents__content")
    
    
    string = ""
    #print(res)
    #print(type(res))
    #print("-------------")
    for link in res:
        string += link.get_text()
      
    return string
    
    
def load_page(url):
    
    driver =webdriver.Firefox(executable_path=r'C:/Users/jazmi/Documents/geckodriver.exe')
    driver.get(url)
    SCROLL_PAUSE_TIME = 30

    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        
    return driver

def scrap_page1(fecha):
    url = "https://elcomercio.pe/archivo/todas/" + fecha +"/"
    driver = load_page(url)
        
    soup = BeautifulSoup(driver.page_source,'html.parser')
    driver.close()
    links = []
    headings = []
    #description = [] 
    for item in soup.find_all('a',class_="story-item__title"):
        headings.append(item.text) # .find('a')['href'])
        links.append("https://elcomercio.pe" + item['href'])
        #print("https://elcomercio.pe" + item['href'])
        #description.append(description_scraper("https://elcomercio.pe" + item['href']))
        #break
    fechas = [str(fecha)] * len(links)
    print(fecha)
    print(len(headings))
    #print(description)
    #return headings, links, fechas, description
    return headings, links, fechas










def scrap_page(fecha):
    url = "https://elcomercio.pe/archivo/todas/" + fecha +"/"
    driver = load_page(url)
        
    soup = BeautifulSoup(driver.page_source,'html.parser')
    driver.close()
    links = []
    headings = []
    for item in soup.find_all('a',class_="story-item__title"):
        headings.append(item.text)
        links.append("https://elcomercio.pe" + item['href'])
    fechas = [str(fecha)] * len(links)
    print(fecha)
    print(len(headings))
    return headings, links, fechas














