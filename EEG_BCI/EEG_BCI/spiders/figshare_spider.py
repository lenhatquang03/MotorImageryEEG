from typing import Any, Iterable
import scrapy
from scrapy import Request
from scrapy.http import Response
from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.action_chains import ActionChains
import time
import os

class FigshareSpider(scrapy.Spider):
    name = "figshare"
    download_dir = "D:/MotorImageryEEG"
    driver = webdriver.Chrome(service = ChromeService(ChromeDriverManager().install()))

    def scroll_to_load(self):
        """ Links to the datasets are loaded dynamically on scroll. """
        actions = ActionChains(self.driver)
        pause_to_load = 2

        while True:
            elements_loaded = len(self.driver.find_elements(By.CSS_SELECTOR, "div.zzfyc"))
            # Scrolling down the page by 10000 pixels
            actions.scroll_by_amount(0, 10000).perform()
            time.sleep(pause_to_load)
            additional_load = len(self.driver.find_elements(By.CSS_SELECTOR, "div.zzfyc"))
            if additional_load == elements_loaded:
                break
        return

    def start_requests(self) -> Iterable[Request]:
        self.driver.get("https://figshare.com/collections/A_large_electroencephalographic_motor_imagery_dataset_for_electroencephalographic_brain_computer_interfaces/3917698/1")

        # Dealing with cookie preference popup
        try:
            accept_cookie = WebDriverWait(self.driver, 5).until(
                ec.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept all')]"))
            )
            accept_cookie.click()
            time.sleep(2)
        except TimeoutException:
            print("Cookie consent not asked for!")
        finally:
            self.scroll_to_load()

        dataset_divs = WebDriverWait(self.driver, 10).until(
            ec.presence_of_all_elements_located((By.CSS_SELECTOR, "div.zzfyc"))
        )

        for div in dataset_divs:
            anchor = div.find_element(By.CSS_SELECTOR, "a.lPlX1._6JpEv")
            dataset_name = anchor.get_attribute("aria-label")
            # Visit pages with datasets only
            if dataset_name.startswith("Experiment"):
                dataset = anchor.get_attribute("href")
                yield scrapy.Request(dataset)
        self.driver.quit()

    def parse(self, response : Response, **kwargs: Any):
        """Downloading .mat files"""
        file_name = response.css("h1.X96Ie::text").get()
        paradigm = file_name.split(" ")[1].split("-")[0]
        paradigm_dir = self.download_dir + f"//{paradigm}"
        os.makedirs(paradigm_dir, exist_ok=True)
        file_path = os.path.join(paradigm_dir, f"{file_name}.mat")
        self.log(f"Downloading file to {file_path}")
        with open(file_path, "wb") as f:
            f.write(response.body)
        self.log(f"Successfully saved {file_path}")