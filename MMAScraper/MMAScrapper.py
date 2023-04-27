import csv
import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options


class FightersStatsScraper:
    def __init__(self):
        self.base_url = "https://www.ufc.com"
        self.start_url = "https://www.ufc.com/athletes/all"
        self.fighter_links = []
        self.session = requests.Session()
        # Create Chrome options
        #chrome_options = Options()
        # Add headless option
        #chrome_options.add_argument("--headless")
        #self.driver = webdriver.Chrome()  # Use appropriate driver for your browser

    def scrape(self):
        self.driver.get(self.start_url)
        wait = WebDriverWait(self.driver, 10)

        while True:
            try:
                load_more_button = wait.until(EC.presence_of_element_located((By.XPATH, '//a[@rel="next"]')))
                self.driver.execute_script("arguments[0].click();", load_more_button)
                time.sleep(2)
                # Wait for content to load, adjust as needed
            except:
                break

        # Extract content after all "Load More" buttons are clicked
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        athletes = soup.select('div[class*="flipcard__action"] a')
        self.fighter_links = [self.base_url + url["href"] for url in athletes]

        # Write fighter_links to a CSV file
        with open('fighter_links.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Fighter Link'])
            writer.writerows([[link] for link in self.fighter_links])

        print(f'{len(self.fighter_links)} fighter links written to CSV.')

    def analyse(self):
        # Scrape fighter bio pages and store data in a list of dictionaries
        fighter_data = []
        with open('fighter_links.csv', 'r') as file:
            csv_reader = csv.reader(file)
            # Skip the header row
            next(csv_reader)
            for row in csv_reader:
                fighter_link = row[0]
                response = self.session.get(fighter_link)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")
                # Extract fighter information from bio page
                fighter_name_elem = soup.select_one('h1.hero-profile__name')
                fighter_name = fighter_name_elem.text.strip() if fighter_name_elem else "N/A"
                fighter_div_elem = soup.select_one("p.hero-profile__division-title")
                fighter_div = fighter_div_elem.text.strip() if fighter_div_elem else "N/A"
                fighter_records_elem = soup.select_one("p.hero-profile__division-body")
                fighter_records = fighter_records_elem.text.strip() if fighter_records_elem else "N/A"
                sig_strikes_elem = soup.select_one("dd.c-overlap__stats-value")
                sig_strikes = sig_strikes_elem.text.strip() if sig_strikes_elem else "N/A"
                print("Name:", fighter_name)
                print("Division:", fighter_div)
                print("Records:", fighter_records)
                print("Striking Accuracy: ")
                print("Sig.Strikes Landed: ", sig_strikes)
                print("Sig.Strikes Attempted: ")
                print("------")
                fighter_data.append({"Name": fighter_name, "Division": fighter_div, "Records": fighter_records})

        return fighter_data

    def run(self):
        fighter_data = self.analyse()
        # Write scraped data to CSV file
        with open("fighters_data.csv", "w", newline="") as csvfile:
            fieldnames = ["Name", "Division", "Records"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(fighter_data)

        print("Data has been saved to fighters_data.csv")


if __name__ == "__main__":
    scraper = FightersStatsScraper()
    scraper.run()
