import requests
from bs4 import BeautifulSoup
import socket
import re

# Configure requests to use Tor
session = requests.Session()
session.proxies = {
    'http': 'socks5h://localhost:9050',
    'https': 'socks5h://localhost:9050'
}

def scrape_and_format(onion_url, output_file):
    try:
        # Step 1: Scrape the onion site
        response = session.get(onion_url)
        response.raise_for_status()  # Check for HTTP errors
        html_content = response.text

        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')
        text_data = soup.get_text()

        # Step 2: Format the scraped data
        formatted_text = clean_text(text_data)

        # Save the formatted data to a file
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(formatted_text)

        print(f"Scraped data saved to {output_file}")
        return output_file

    except (requests.RequestException, socket.error) as e:
        print(f"Error accessing {onion_url}: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def clean_text(text):
    # Remove non-printable characters
    text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)
   
    # Normalize whitespace: replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
   
    # Strip leading and trailing whitespace
    text = text.strip()
   
    return text

def main():
    onion_url = "http://wms5y25kttgihs4rt2sifsbwsjqjrx3vtc42tsu2obksqkj7y666fgid.onion/"  # Replace with your onion link
    output_file = "formatted_data1.txt"  # Replace with your desired output file name

    # Scrape and format the onion site data
    formatted_data = scrape_and_format(onion_url, output_file)
    if formatted_data is None:
        print("Failed to scrape and format the site data.")
        return

if __name__ == "__main__":
    main()


