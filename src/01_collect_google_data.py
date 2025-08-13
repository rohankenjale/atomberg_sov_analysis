import os, time, random, logging
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

BRANDS = ["atomberg","orient","havells","crompton","polycab"]

def canonicalize_url(u):
    try:
        p = urlparse(u)
        return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))
    except Exception:
        return u

def _setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    )
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    wait = WebDriverWait(driver, 10)
    return driver, wait

def _collect_one_query(driver, wait, query, n_results):
    collected = []
    start = 0
    abs_rank = 0

    while len(collected) < n_results:
        url = f"https://www.google.com/search?q={query.replace(' ','+')}&hl=en&gl=IN&pws=0&num=10&start={start}"
        logging.info(f"[{query}] Loading: {url}")
        driver.get(url)

        try:
            consent = wait.until(EC.presence_of_all_elements_located(
                (By.XPATH, "//*[contains(., 'I agree') or contains(., 'Accept all')]")
            ))
            for el in consent:
                try:
                    if el.is_displayed() and el.tag_name.lower() in ("button","div","span"):
                        el.click()
                        break
                except Exception:
                    pass
        except Exception:
            pass

        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#search")))
        except Exception:
            logging.warning(f"[{query}] #search not found; breaking.")
            break

        cards = driver.find_elements(By.CSS_SELECTOR, "#search div.MjjYud")
        page_rank = 0

        for c in cards:
            try:
                h = c.find_element(By.CSS_SELECTOR, "h3")
            except Exception:
                continue

            a = None
            try:
                a = h.find_element(By.XPATH, "./ancestor::a[1]")
            except Exception:
                try:
                    a = c.find_element(By.CSS_SELECTOR, "a")
                except Exception:
                    pass
            if not a:
                continue

            href = a.get_attribute("href")
            if not href or not href.startswith("http"):
                continue
            if "/aclk?" in href or "google.com/aclk" in href:
                continue

            title = h.text.strip()
            desc = ""
            for sel in ["div.VwiC3b", "div.yXK7lf"]:
                try:
                    desc = c.find_element(By.CSS_SELECTOR, sel).text.strip()
                    if desc:
                        break
                except Exception:
                    pass

            page_rank += 1
            abs_rank += 1
            text = f"{title} {desc}".lower()
            mentions = {f"mention_{b}": (b in text) for b in BRANDS}

            collected.append({
                "query": query,
                "page": start//10 + 1,
                "rank_page": page_rank,
                "rank_abs": abs_rank,
                "title": title,
                "url": canonicalize_url(href),
                "description": desc,
                "platform": "google",
                "result_type": "organic",
                "engagement": 0,  
                **mentions
            })

            if len(collected) >= n_results:
                break

        if len(collected) < n_results:
            # Next page or stop
            try:
                next_btn = driver.find_element(By.ID, "pnnext")
                next_btn.click()
                start += 10
                time.sleep(random.uniform(5.0, 10.0))
            except Exception:
                logging.info(f"[{query}] No next page; stopping.")
                break

        time.sleep(random.uniform(3.0, 6.0))

    return collected

def collect_google_multi(queries, n_results_per_query=20, out_dir="data/raw", out_name="google_multi_queries"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    driver, wait = _setup_driver()
    all_rows = []
    try:
        for q in queries:
            rows = _collect_one_query(driver, wait, q, n_results_per_query)
            logging.info(f"[{q}] Collected {len(rows)} results.")
            all_rows.extend(rows)
            time.sleep(random.uniform(2.0, 4.0))
    finally:
        driver.quit()

    if not all_rows:
        logging.warning("No results collected for any query.")
        return None

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["url","title","query"])
    csv_path = os.path.join(out_dir, f"{out_name}.csv")
    parquet_path = os.path.join(out_dir, f"{out_name}.parquet")
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    logging.info(f"Saved {len(df)} rows to:\n  {csv_path}\n  {parquet_path}")
    return df

if __name__ == "__main__":
    QUERIES = ["smart fan", "energy efficient fan", "BLDC fan"]
    collect_google_multi(QUERIES, n_results_per_query=20, out_dir="data/raw", out_name="google_sov_india")
