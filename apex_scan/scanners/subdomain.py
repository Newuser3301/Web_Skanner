from ..deps import *

class PurePythonSubdomainEnumerator:
    """Pure Python subdomain enumerator"""
    
    def __init__(self):
        self.wordlist = self.load_wordlist()
        self.resolvers = [
            "8.8.8.8",  # Google DNS
            "1.1.1.1",  # Cloudflare DNS
            "9.9.9.9",  # Quad9
            "208.67.222.222"  # OpenDNS
        ]
    
    def load_wordlist(self) -> List[str]:
        """Load subdomain wordlist"""
        common_subdomains = [
            "www", "mail", "ftp", "admin", "webmail", "smtp", "pop", "imap",
            "test", "dev", "development", "staging", "prod", "production",
            "api", "rest", "graphql", "soap", "xmlrpc",
            "blog", "news", "forum", "community", "support",
            "static", "assets", "cdn", "media", "images", "uploads",
            "app", "apps", "application", "portal", "dashboard",
            "secure", "auth", "authentication", "login", "signin",
            "db", "database", "sql", "mysql", "postgres", "mongodb",
            "redis", "cache", "memcache", "elasticsearch", "kibana",
            "jenkins", "git", "gitlab", "github", "bitbucket",
            "docker", "kubernetes", "k8s", "helm", "istio",
            "prometheus", "grafana", "alertmanager", "thanos",
            "vpn", "proxy", "bastion", "jump", "gateway",
            "ns1", "ns2", "ns3", "ns4", "dns", "bind",
            "mx", "mx1", "mx2", "mx3", "mailin", "mailout",
            "owa", "exchange", "outlook", "sharepoint",
            "crm", "erp", "hr", "payroll", "accounting",
            "sales", "marketing", "support", "helpdesk",
            "monitor", "monitoring", "nagios", "zabbix",
            "backup", "backups", "archive", "archives"
        ]
        return common_subdomains
    
    async def enumerate(self, domain: str, use_wordlist: bool = True, 
                       use_cert_transparency: bool = True,
                       use_search_engines: bool = True) -> Dict[str, Any]:
        """Enumerate subdomains"""
        
        results = {
            "domain": domain,
            "subdomains": set(),
            "methods": {
                "dns_bruteforce": [],
                "cert_transparency": [],
                "search_engines": []
            },
            "ips": defaultdict(list),
            "cnames": {}
        }
        
        # Method 1: DNS Bruteforce
        if use_wordlist:
            dns_results = await self.dns_bruteforce(domain)
            results["subdomains"].update(dns_results)
            results["methods"]["dns_bruteforce"] = list(dns_results)
        
        # Method 2: Certificate Transparency
        if use_cert_transparency:
            ct_results = await self.certificate_transparency(domain)
            results["subdomains"].update(ct_results)
            results["methods"]["cert_transparency"] = list(ct_results)
        
        # Method 3: Search Engines (simplified)
        if use_search_engines:
            search_results = await self.search_engine_dorking(domain)
            results["subdomains"].update(search_results)
            results["methods"]["search_engines"] = list(search_results)
        
        # Resolve IPs and CNAMEs
        for subdomain in results["subdomains"]:
            ip_addresses = await self.resolve_ip(subdomain)
            cname = await self.resolve_cname(subdomain)
            
            if ip_addresses:
                results["ips"][subdomain] = ip_addresses
            
            if cname:
                results["cnames"][subdomain] = cname
        
        # Convert set to list for JSON serialization
        results["subdomains"] = list(results["subdomains"])
        
        return results
    
    async def dns_bruteforce(self, domain: str) -> Set[str]:
        """Bruteforce subdomains using DNS"""
        found = set()
        
        async def check_subdomain(subdomain: str):
            full_domain = f"{subdomain}.{domain}"
            
            for resolver in self.resolvers:
                try:
                    answers = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: dns.resolver.resolve(full_domain, 'A')
                    )
                    
                    if answers:
                        found.add(full_domain)
                        break
                
                except dns.resolver.NXDOMAIN:
                    pass
                except dns.resolver.NoAnswer:
                    pass
                except Exception:
                    pass
        
        # Check common subdomains
        tasks = [check_subdomain(sub) for sub in self.wordlist]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return found
    
    async def certificate_transparency(self, domain: str) -> Set[str]:
        """Get subdomains from Certificate Transparency logs"""
        found = set()
        
        try:
            # Use crt.sh API
            url = f"https://crt.sh/?q=%25.{domain}&output=json"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for entry in data:
                            name = entry.get("name_value", "")
                            
                            # Extract subdomains
                            names = name.split('\n')
                            for n in names:
                                n = n.strip().lower()
                                if domain in n:
                                    found.add(n)
        
        except Exception as e:
            pass
        
        return found
    
    async def search_engine_dorking(self, domain: str) -> Set[str]:
        """Get subdomains from search engines"""
        found = set()
        
        # This is a simplified version
        # Real implementation would use search engine APIs
        
        # Common patterns
        patterns = [
            f"site:*.{domain}",
            f"inurl:{domain}",
            f"intitle:{domain}"
        ]
        
        # Use local DNS enumeration as fallback
        try:
            # Try DNS zone transfer (unlikely to work but worth trying)
            transfer = await self.dns_zone_transfer(domain)
            found.update(transfer)
        except:
            pass
        
        return found
    
    async def dns_zone_transfer(self, domain: str) -> Set[str]:
        """Attempt DNS zone transfer"""
        found = set()
        
        # Common nameservers
        ns_servers = [f"ns1.{domain}", f"ns2.{domain}", f"dns1.{domain}"]
        
        for ns in ns_servers:
            try:
                # Try to resolve nameserver first
                ns_ip = await self.resolve_ip(ns)
                if not ns_ip:
                    continue
                
                # Attempt zone transfer
                transfer = dns.zone.from_xfr(dns.query.xfr(ns_ip[0], domain))
                if transfer:
                    for name in transfer.nodes.keys():
                        full_name = f"{name}.{domain}"
                        found.add(full_name)
            
            except:
                continue
        
        return found
    
    async def resolve_ip(self, domain: str) -> List[str]:
        """Resolve domain to IP addresses"""
        ips = []
        
        try:
            answers = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: dns.resolver.resolve(domain, 'A')
            )
            
            for answer in answers:
                ips.append(answer.address)
        
        except:
            pass
        
        return ips
    
    async def resolve_cname(self, domain: str) -> Optional[str]:
        """Resolve CNAME record"""
        try:
            answers = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: dns.resolver.resolve(domain, 'CNAME')
            )
            
            if answers:
                return str(answers[0].target)
        
        except:
            pass
        
        return None
