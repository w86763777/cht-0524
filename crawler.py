import abc
import argparse
import os
import time
from collections import defaultdict
from typing import List, Dict


import hanzidentifier
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


class ParseFailedException(Exception):
    pass


class Parser(abc.ABC):
    source: str

    @abc.abstractmethod
    def parse(self, response: requests.Response) -> str:
        pass


class NewTalkParser(Parser):
    source: str = "NewTalk.tw"

    def parse(self, response: requests.Response) -> str:
        soup = BeautifulSoup(response.text, 'html.parser')
        article_body = soup.find('div', class_='articleBody')
        content = ""
        if article_body is not None:
            for p in article_body.find_all('p', recursive=False):
                if p is not None:
                    content += p.get_text(strip=True)
        if content == "":
            raise ParseFailedException("Failed to parse the content.")
        return content


class TaronewsParser(Parser):
    source: str = "芋傳媒"

    def parse(self, response: requests.Response) -> str:
        soup = BeautifulSoup(response.text, 'html.parser')
        article_body = soup.select_one('article>div.entry-content')
        if article_body is None:
            raise ParseFailedException("Failed to parse the content.")
        else:
            return article_body.get_text(strip=True)


class StormParser(Parser):
    source: str = "風傳媒"

    def parse(self, response: requests.Response) -> str:
        soup = BeautifulSoup(response.text, 'html.parser')
        content = ""
        paragraphs = soup.select('article p[aid]')
        for p in paragraphs:
            content += p.get_text(strip=True)
        if len(content) == 0:
            raise ParseFailedException("Failed to parse the content.")
        else:
            return content


def append_to_csv(file_name: str, data: Dict[str, List[str]]):
    df = pd.DataFrame.from_dict(data)
    if os.path.exists(file_name):
        # Append without header
        df.to_csv(file_name, mode='a', index=False, header=False)
    else:
        # Write with header
        df.to_csv(file_name, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help="Path to the csv file.")
    parser.add_argument('--out', type=str, required=True,
                        help="Path to the output csv file.")
    parser.add_argument('--max-retry', type=int, default=3,
                        help="Max retry times for each url.")
    parser.add_argument('--delay', type=float, default=0.5,
                        help="Delay time between each request in seconds.")
    args = parser.parse_args()

    response_parsers = [
        NewTalkParser(),
        # TaronewsParser(),
        StormParser()
    ]
    response_parsers = {
        parser.source: parser for parser in response_parsers
    }
    supported_sources = set(response_parsers.keys())

    df = pd.read_csv(args.data, usecols=['title', 'desc', 'source', 'url', 'date'])
    print(f"# news: {len(df):6d}")
    df = df.dropna()
    print(f"# news: {len(df):6d} (dropping NaN)")
    df = df[df.apply(lambda row: hanzidentifier.is_traditional(row['title']), axis=1)]
    print(f"# news: {len(df):6d} (filtering traditional Chinese)")
    df = df[df.apply(lambda row: row['source'] in supported_sources, axis=1)]
    print(f"# news: {len(df):6d} (filtering supported sources {supported_sources})")

    if os.path.exists(args.out):
        cached = set(pd.read_csv(args.out, usecols=['url'])['url'].values)
    else:
        cached = set()

    data = defaultdict(list)
    with tqdm(df.iterrows(), total=len(df), desc="Crawling") as pbar:
        for _, row in pbar:
            # Skip if already cached
            if row['url'] in cached:
                continue
            else:
                start = time.time()
                # Initialize data
                for name, value in row.items():
                    data[name].append(value)
                # Start sending requests
                for _ in range(args.max_retry):
                    response = None
                    try:
                        response = requests.get(row['url'], cookies={'over18': '1'})
                        if response.status_code == 200:
                            break
                    except Exception:
                        time.sleep(1)
                if response is None:
                    # Skip if failed to GET
                    pbar.write(f"[GET Err] {row['url']}")
                    data['content'].append("")
                elif response.status_code != 200:
                    # Skip if status code is not 200
                    pbar.write(f"[Not 200] {row['url']}")
                    data['content'].append("")
                else:
                    # Parse the response
                    try:
                        content = response_parsers[row['source']].parse(response)
                        elapsed = time.time() - start
                        pbar.write(f"[Success] {row['url']} ({elapsed:.2f}s)")
                        data['content'].append(content)
                    except ParseFailedException:
                        pbar.write(f"[Parse:(] {row['url']}")
                        data['content'].append("")

                # flush
                if len(data['content']) >= 100:
                    append_to_csv(args.out, data)
                    data = defaultdict(list)
                    pbar.write("Flush")

                time.sleep(args.delay)

    # flush
    append_to_csv(args.out, data)


if __name__ == '__main__':
    main()
