import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import time
import os
from typing import Set, List, Optional, Dict

class PoliteCrawler:
    def __init__(self, start_url: str, max_depth: int = 3, save_dir: str = "crawled_data"):
        self.start_url = start_url
        self.max_depth = max_depth
        self.save_dir = save_dir
        self.visited: Set[str] = set()
        self.robot_parser: Optional[RobotFileParser] = None
        self.base_domain = urlparse(start_url).netloc
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MyPoliteCrawler/1.0 (+https://yourwebsite.com/contact)"
        })

    def init_robots_parser(self):
        if self.robot_parser is not None:
            return
        self.robot_parser = RobotFileParser()
        robots_url = urljoin(self.start_url, "/robots.txt")
        try:
            self.robot_parser.set_url(robots_url)
            self.robot_parser.read()
            print(f"robots.txt 読み込み完了")
        except:
            print("robots.txt なし → すべて許可")
            self.robot_parser = None

    def can_fetch(self, url: str) -> bool:
        if self.robot_parser is None:
            return True
        try:
            return self.robot_parser.can_fetch(self.session.headers["User-Agent"], url)
        except:
            return True

    def save_content(self, url: str, html: str, text: str) -> Dict[str, str]:
        """UTF-8 with BOMで確実に保存"""
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "html"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "text"), exist_ok=True)
        
        safe_name = url.replace("https://", "").replace("http://", "").replace("/", "_").strip("_") or "index"
        
        html_path = os.path.join(self.save_dir, "html", f"{safe_name}.html")
        text_path = os.path.join(self.save_dir, "text", f"{safe_name}.txt")
        
        # BOM付きUTF-8で保存（VSCodeが認識しやすい）
        with open(html_path, "w", encoding="utf-8-sig") as f:
            f.write(html)
        
        with open(text_path, "w", encoding="utf-8-sig") as f:
            f.write(text)
        
        print(f"💾 保存完了: {safe_name}")
        return {"html": html_path, "text": text_path}

    def fetch_urls(self, url: str, depth: int = 0, collected: Optional[Set[str]] = None) -> Set[str]:
        if collected is None:
            collected = set()
        
        if depth > self.max_depth or url in self.visited:
            return collected
        
        if not self.can_fetch(url):
            print(f"robots.txtによりスキップ: {url}")
            return collected

        self.visited.add(url)
        collected.add(url)
        
        try:
            print(f"取得中 (深さ {depth}): {url}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            # === ここを大幅強化：文字エンコーディング対策 ===
            # 1. レスポンスの推定エンコーディングを使う
            if response.encoding == 'ISO-8859-1':  # requestsのデフォルトがおかしい場合
                response.encoding = response.apparent_encoding  # chardetなどで推定
            
            html = response.text
            
            # BeautifulSoupでさらに正しく解析
            soup = BeautifulSoup(html, "html.parser")
            
            # 不要タグ除去
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()
            
            text = soup.get_text(separator="\n", strip=True)
            
            # 保存
            self.save_content(url, html, text)
            
            if depth == self.max_depth:
                return collected
                
            # リンク追跡
            for link in soup.find_all("a", href=True):
                next_url = urljoin(url, link["href"])
                if (urlparse(next_url).netloc == self.base_domain and 
                    next_url not in self.visited):
                    time.sleep(1.5)  # 少し長めに
                    self.fetch_urls(next_url, depth + 1, collected)
                    
        except Exception as e:
            print(f"エラー ({url}): {e}")
        
        return collected

    def get_all_urls(self) -> List[str]:
        """クローリング実行 → URLリストを返す"""
        self.init_robots_parser()
        print(f"クローリング開始: {self.start_url}")
        
        collected = self.fetch_urls(self.start_url)
        url_list = sorted(list(collected))
        
        print(f"完了！ 合計 {len(url_list)} 件")
        return url_list
