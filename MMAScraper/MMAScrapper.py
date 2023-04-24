import csv
import time
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
        # Create Chrome options
        chrome_options = Options()
        # Add headless option
        chrome_options.add_argument("--headless")
        self.driver = webdriver.Chrome()  # Use appropriate driver for your browser

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

        # Scrape fighter bio pages and store data in a list of dictionaries
        fighter_data = []
        for fighter_link in self.fighter_links:
            self.driver.get(fighter_link)
            soup = BeautifulSoup(self.driver.page_source, "html.parser")

            # Extract fighter information from bio page
            fighter_name_elem = soup.select_one('h1.hero-profile__name')
            fighter_name = fighter_name_elem.text.strip() if fighter_name_elem else "N/A"
            fighter_div_elem = soup.select_one("p.hero-profile__division-title")
            fighter_div = fighter_div_elem.text.strip() if fighter_div_elem else "N/A"
            fighter_records_elem = soup.select_one("p.hero-profile__division-body")
            fighter_records = fighter_records_elem.text.strip() if fighter_records_elem else "N/A"
            fighter_data.append({"Name": fighter_name, "Division": fighter_div, "Records": fighter_records})

        return fighter_data

    def run(self):
        fighter_data = self.scrape()
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
