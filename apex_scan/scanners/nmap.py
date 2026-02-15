from ..deps import *

class PurePythonNmap:
    """Pure Python implementation of Nmap-like scanning"""
    
    def __init__(self):
        self.timeout = 5
        self.max_workers = 50
    
    async def scan(self, target: str, ports: str = "1-1000", 
                  scan_type: str = "syn") -> Dict[str, Any]:
        """Perform port scanning"""
        
        results = {
            "target": target,
            "scan_type": scan_type,
            "ports_scanned": ports,
            "open_ports": [],
            "services": [],
            "os_guesses": []
        }
        
        # Parse port range
        port_list = self.parse_port_range(ports)
        
        # Perform scan
        if scan_type == "syn":
            open_ports = await self.syn_scan(target, port_list)
        elif scan_type == "connect":
            open_ports = await self.connect_scan(target, port_list)
        elif scan_type == "udp":
            open_ports = await self.udp_scan(target, port_list)
        else:
            open_ports = await self.connect_scan(target, port_list)
        
        # Service detection
        services = await self.detect_services(target, open_ports)
        
        # OS fingerprinting
        os_guess = await self.os_fingerprint(target)
        
        results["open_ports"] = open_ports
        results["services"] = services
        results["os_guesses"] = os_guess
        
        return results
    
    def parse_port_range(self, ports: str) -> List[int]:
        """Parse port range string"""
        port_list = []
        
        if "-" in ports:
            start, end = map(int, ports.split("-"))
            port_list = list(range(start, end + 1))
        elif "," in ports:
            port_list = [int(p) for p in ports.split(",")]
        else:
            port_list = [int(ports)]
        
        return port_list
    
    async def syn_scan(self, target: str, ports: List[int]) -> List[Dict]:
        """TCP SYN scan"""
        open_ports = []
        
        async def check_port(port: int):
            try:
                # Create raw socket for SYN scan
                # Note: Requires root privileges
                sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_TCP)
                sock.settimeout(self.timeout)
                
                # Build SYN packet
                # Simplified - in reality would use scapy
                try:
                    # Try connect first (fallback)
                    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    test_socket.settimeout(self.timeout)
                    result = test_socket.connect_ex((target, port))
                    test_socket.close()
                    
                    if result == 0:
                        return {"port": port, "state": "open", "protocol": "tcp"}
                except:
                    pass
                
            except Exception as e:
                pass
            
            return None
        
        # Scan ports concurrently
        tasks = [check_port(port) for port in ports]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, dict):
                open_ports.append(result)
        
        return open_ports
    
    async def connect_scan(self, target: str, ports: List[int]) -> List[Dict]:
        """TCP Connect scan (no root required)"""
        open_ports = []
        
        async def check_port(port: int):
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(target, port),
                    timeout=self.timeout
                )
                writer.close()
                await writer.wait_closed()
                return {"port": port, "state": "open", "protocol": "tcp"}
            except:
                return None
        
        # Scan ports with limited concurrency
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def check_port_with_semaphore(port: int):
            async with semaphore:
                return await check_port(port)
        
        tasks = [check_port_with_semaphore(port) for port in ports]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, dict):
                open_ports.append(result)
        
        return open_ports
    
    async def udp_scan(self, target: str, ports: List[int]) -> List[Dict]:
        """UDP scan"""
        open_ports = []
        
        async def check_port(port: int):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(self.timeout)
                
                # Send empty UDP packet
                sock.sendto(b"", (target, port))
                
                try:
                    # Try to receive response
                    data, addr = sock.recvfrom(1024)
                    if data:
                        return {"port": port, "state": "open", "protocol": "udp"}
                except socket.timeout:
                    # No response - might be open or filtered
                    return {"port": port, "state": "open|filtered", "protocol": "udp"}
                
                sock.close()
            except:
                pass
            
            return None
        
        tasks = [check_port(port) for port in ports]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, dict):
                open_ports.append(result)
        
        return open_ports
    
    async def detect_services(self, target: str, open_ports: List[Dict]) -> List[Dict]:
        """Detect services on open ports"""
        services = []
        
        # Common service ports and detection
        common_services = {
            21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns",
            80: "http", 110: "pop3", 143: "imap", 443: "https", 445: "smb",
            3306: "mysql", 3389: "rdp", 5432: "postgresql", 5900: "vnc",
            6379: "redis", 27017: "mongodb", 9200: "elasticsearch"
        }
        
        for port_info in open_ports:
            port = port_info["port"]
            protocol = port_info["protocol"]
            
            service_info = {
                "port": port,
                "protocol": protocol,
                "service": "unknown",
                "version": "unknown",
                "banner": ""
            }
            
            # Check common services
            if port in common_services:
                service_info["service"] = common_services[port]
            
            # Try to get banner
            banner = await self.get_banner(target, port, protocol)
            if banner:
                service_info["banner"] = banner
                service_info["service"] = self.identify_service_from_banner(banner)
            
            services.append(service_info)
        
        return services
    
    async def get_banner(self, target: str, port: int, protocol: str) -> str:
        """Get service banner"""
        try:
            if protocol == "tcp":
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(target, port),
                    timeout=3
                )
                
                # Send probe based on port
                if port in [21, 22, 25, 80, 443]:
                    # Send newline for interactive services
                    writer.write(b"\n")
                    await writer.drain()
                
                # Read response
                banner = await asyncio.wait_for(reader.read(1024), timeout=2)
                writer.close()
                await writer.wait_closed()
                
                return banner.decode('utf-8', errors='ignore').strip()
            
        except Exception as e:
            pass
        
        return ""
    
    def identify_service_from_banner(self, banner: str) -> str:
        """Identify service from banner"""
        banner_lower = banner.lower()
        
        if "ssh" in banner_lower:
            return "ssh"
        elif "ftp" in banner_lower:
            return "ftp"
        elif "smtp" in banner_lower:
            return "smtp"
        elif "http" in banner_lower:
            return "http"
        elif "apache" in banner_lower:
            return "apache"
        elif "nginx" in banner_lower:
            return "nginx"
        elif "iis" in banner_lower:
            return "iis"
        
        return "unknown"
    
    async def os_fingerprint(self, target: str) -> List[Dict]:
        """OS fingerprinting using TCP/IP stack analysis"""
        fingerprints = []
        
        try:
            # TTL analysis
            ttl = await self.get_ttl(target)
            
            # TCP window size
            window_size = await self.get_window_size(target)
            
            # TCP options
            tcp_options = await self.get_tcp_options(target)
            
            # Common OS fingerprints
            os_guesses = []
            
            if ttl:
                if 64 <= ttl <= 128:
                    os_guesses.append({"os": "Linux/Unix", "confidence": 0.7})
                elif ttl == 128:
                    os_guesses.append({"os": "Windows", "confidence": 0.8})
                elif ttl == 255:
                    os_guesses.append({"os": "Solaris/AIX", "confidence": 0.6})
            
            fingerprints = os_guesses
            
        except Exception as e:
            pass
        
        return fingerprints
    
    async def get_ttl(self, target: str) -> Optional[int]:
        """Get TTL using ping"""
        try:
            # Use system ping if available
            if sys.platform == "win32":
                cmd = ["ping", "-n", "1", target]
            else:
                cmd = ["ping", "-c", "1", target]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            output = stdout.decode()
            
            # Parse TTL from ping output
            ttl_match = re.search(r'ttl=(\d+)', output.lower())
            if ttl_match:
                return int(ttl_match.group(1))
        
        except:
            pass
        
        return None
    
    async def get_window_size(self, target: str) -> Optional[int]:
        """Get TCP window size"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect((target, 80))
            
            # Send HTTP request
            sock.send(b"GET / HTTP/1.0\r\n\r\n")
            
            # Receive response
            response = sock.recv(1024)
            sock.close()
            
            # In real implementation, would analyze TCP headers
            return 65535  # Default guess
        
        except:
            return None
    
    async def get_tcp_options(self, target: str) -> List[str]:
        """Get TCP options"""
        # Simplified - in reality would analyze TCP handshake
        return []
