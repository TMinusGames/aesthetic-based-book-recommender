import hashlib, io, requests
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from pathlib import Path
from PIL import Image

# Created by Jasper van der Valk (UvAnetID: 13854577) for Bachelor Thesis: Matching atmospheric descriptors in image and text: What are internet aesthetics anyway?


options = ChromeOptions()
options.add_argument("--headless=new")
driver = webdriver.Chrome(options=options)

#driver.get("https://aesthetics.fandom.com/wiki/Dark_Academia")
#driver.get("https://aesthetics.fandom.com/wiki/Cottagecore")
#driver.get("https://aesthetics.fandom.com/wiki/Angelcore")
#driver.get("https://aesthetics.fandom.com/wiki/Spacecore")
#driver.get("https://aesthetics.fandom.com/wiki/Soft_Grunge")
#driver.get("https://aesthetics.fandom.com/wiki/Dopamine")
#driver.get("https://aesthetics.fandom.com/wiki/Lovecore")
#driver.get("https://aesthetics.fandom.com/wiki/Kidcore")
#driver.get("https://aesthetics.fandom.com/wiki/Kawaii")
driver.get("https://aesthetics.fandom.com/wiki/Steampunk")
results = []
content = driver.page_source
soup = BeautifulSoup(content, "html.parser")
# print the webscraped result to find which tag you need for the images
print(soup)
driver.quit()


def parse_image_urls(classes, location, source):
    for a in soup.find_all(attrs={"class": classes}):
        print(a)
        name = a.find(location)
        if name not in results:
            link = name.get(source)
            link = link[:link.index("latest") + len("latest")]
            results.append(link)
    return results


if __name__ == "__main__":
    # Rotate between these two if one does not work
    #returned_results = parse_image_urls("gallery-image-wrapper accent", "img", "src")
    returned_results = parse_image_urls("gallery-image-wrapper", "img", "src")
    for b in returned_results:
        image_content = requests.get(b).content
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert("RGB")
        file_path = Path("/home/jaspervdvalk/Documents/Bachelor Thesis/aesthetics/steampunk", hashlib.sha1(image_content).hexdigest()[:10] + ".png")
        image.save(file_path, "PNG", quality=80)
