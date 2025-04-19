import requests
from bs4 import BeautifulSoup
import pandas as pd

def worldbankimport(countries, indicators):
    country_str = ";".join(countries)
    indicator_str = ";".join(indicators)
    base_url = "https://api.worldbank.org/v2/country/"
    url = f"{base_url}{country_str}/indicator/{indicator_str}?source=2"

    xmldf = pd.DataFrame()
    response = requests.get(f"{url}&page=1")
    soup = BeautifulSoup(response.content, 'xml')
    totalpages = int(soup.find("data")["pages"])  # Number of pages

    for page in range(1, totalpages + 1):  # Loop through all pages
        response = requests.get(f"{url}&page={page}")
        soup = BeautifulSoup(response.content, 'xml')
        catalog = soup.contents[0]
        # Parse all <wb:data> elements
        for i in range(len(catalog.contents)):
            row = catalog.contents[i]
            if not row.name:
                continue
            row_dict = {}
            cells = row.find_all(True,recursive=False)

            for cell in cells:
                if cell.name == "indicator" or cell.name=="country":
                    row_dict[f"{cell.name}_id"] = cell.get("id", None)
                    row_dict[f"{cell.name}_value"] = cell.string
                else:
                    row_dict[cell.name] = cell.string
            xmldf = pd.concat([xmldf, pd.DataFrame(row_dict,index=[0])], ignore_index=True)
    return xmldf


# Example usage
countries = ["CHN", "USA"]
indicators = ["NY.GDP.PCAP.CD"]
xmldf = worldbankimport(countries, indicators)
print(xmldf)