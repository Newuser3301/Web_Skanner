from ..deps import *

class PurePythonDirBuster:
    """Pure Python directory bruteforcer"""
    
    def __init__(self):
        self.common_dirs = self.load_common_directories()
        self.common_files = self.load_common_files()
        self.extensions = [".php", ".html", ".txt", ".json", ".xml", ".asp", ".aspx", ".jsp"]
    
    def load_common_directories(self) -> List[str]:
        """Load common directory names"""
        return [
            "admin", "administrator", "login", "panel", "dashboard",
            "wp-admin", "wp-content", "wp-includes", "phpmyadmin",
            "test", "backup", "backups", "old", "temp", "tmp",
            "api", "rest", "graphql", "soap", "xmlrpc",
            "config", "configuration", "settings", "setup",
            "uploads", "files", "images", "assets", "static",
            "private", "secret", "hidden", "secure",
            "cgi-bin", "cgi", "bin", "scripts",
            ".git", ".svn", ".hg", ".env", ".well-known"
        ]
    
    def load_common_files(self) -> List[str]:
        """Load common file names"""
        return [
            "index", "main", "home", "default", "config",
            "configuration", "settings", "setup", "install",
            "admin", "administrator", "login", "logout", "register",
            "robots.txt", "sitemap.xml", "crossdomain.xml",
            ".htaccess", ".htpasswd", "web.config", "phpinfo.php",
            "test.php", "info.php", "debug.php", "console.php",
            "api.php", "api.json", "api.xml", "rest.php",
            "backup.zip", "backup.sql", "dump.sql", "database.sql",
            "readme.txt", "license.txt", "changelog.txt"
        ]
    
    async def scan(self, base_url: str, wordlist: Optional[List[str]] = None, 
                  extensions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Scan for directories and files"""
        
        if wordlist is None:
            wordlist = self.common_dirs + self.common_files
        
        if extensions is None:
            extensions = self.extensions
        
        results = {
            "base_url": base_url,
            "directories_found": [],
            "files_found": [],
            "status_codes": defaultdict(int),
            "total_requests": 0
        }
        
        # Ensure base URL ends with /
        if not base_url.endswith('/'):
            base_url += '/'
        
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(20)  # 20 concurrent requests
        
        async def check_path(path: str, is_file: bool = False):
            async with semaphore:
                url = base_url + path
                
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, timeout=10) as response:
                            status = response.status
                            results["status_codes"][status] += 1
                            results["total_requests"] += 1
                            
                            if status in [200, 301, 302, 403]:
                                result = {
                                    "url": url,
                                    "status": status,
                                    "path": path,
                                    "type": "file" if is_file else "directory",
                                    "size": response.headers.get('Content-Length', 0),
                                    "content_type": response.headers.get('Content-Type', '')
                                }
                                
                                if is_file:
                                    results["files_found"].append(result)
                                else:
                                    results["directories_found"].append(result)
                
                except Exception as e:
                    pass
        
        # Check directories
        dir_tasks = [check_path(dir_name, False) for dir_name in self.common_dirs]
        
        # Check files with extensions
        file_tasks = []
        for file_name in self.common_files:
            # Check without extension
            file_tasks.append(check_path(file_name, True))
            
            # Check with extensions
            for ext in extensions:
                file_tasks.append(check_path(f"{file_name}{ext}", True))
        
        # Run all checks
        all_tasks = dir_tasks + file_tasks
        await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Sort results by status code
        results["directories_found"].sort(key=lambda x: x["status"])
        results["files_found"].sort(key=lambda x: x["status"])
        
        return results
