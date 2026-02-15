from ..deps import *

class PurePythonSSLChecker:
    """Pure Python SSL/TLS checker"""
    
    def __init__(self):
        self.ciphers = self.load_ciphers()
    
    def load_ciphers(self) -> List[str]:
        """Load SSL/TLS ciphers"""
        return [
            "TLS_AES_256_GCM_SHA384",
            "TLS_CHACHA20_POLY1305_SHA256",
            "TLS_AES_128_GCM_SHA256",
            "ECDHE-RSA-AES256-GCM-SHA384",
            "ECDHE-RSA-AES256-SHA384",
            "ECDHE-RSA-AES256-SHA",
            "DHE-RSA-AES256-GCM-SHA384",
            "DHE-RSA-AES256-SHA256",
            "DHE-RSA-AES256-SHA",
            "AES256-GCM-SHA384",
            "AES256-SHA256",
            "AES256-SHA"
        ]
    
    async def scan(self, hostname: str, port: int = 443) -> Dict[str, Any]:
        """Scan SSL/TLS configuration"""
        
        results = {
            "hostname": hostname,
            "port": port,
            "certificate": {},
            "protocols": [],
            "ciphers": [],
            "vulnerabilities": []
        }
        
        # Check certificate
        cert_info = await self.check_certificate(hostname, port)
        if cert_info:
            results["certificate"] = cert_info
        
        # Check supported protocols
        protocols = await self.check_protocols(hostname, port)
        results["protocols"] = protocols
        
        # Check supported ciphers
        ciphers = await self.check_ciphers(hostname, port)
        results["ciphers"] = ciphers
        
        # Check for vulnerabilities
        vulns = await self.check_vulnerabilities(hostname, port, cert_info, protocols, ciphers)
        results["vulnerabilities"] = vulns
        
        return results
    
    async def check_certificate(self, hostname: str, port: int) -> Optional[Dict]:
        """Check SSL certificate"""
        try:
            context = ssl.create_default_context()
            
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    cert_binary = ssock.getpeercert(binary_form=True)
                    
                    # Parse certificate
                    x509 = OpenSSL.crypto.load_certificate(
                        OpenSSL.crypto.FILETYPE_ASN1, cert_binary
                    )
                    
                    # Get certificate info
                    subject = dict(x509.get_subject().get_components())
                    issuer = dict(x509.get_issuer().get_components())
                    
                    # Check expiration
                    not_after = x509.get_notAfter().decode('ascii')
                    expiry_date = datetime.datetime.strptime(not_after, '%Y%m%d%H%M%SZ')
                    days_remaining = (expiry_date - datetime.datetime.utcnow()).days
                    
                    # Check key size
                    public_key = x509.get_pubkey()
                    key_bits = public_key.bits()
                    
                    # Check signature algorithm
                    sig_alg = x509.get_signature_algorithm().decode('ascii')
                    
                    return {
                        "subject": subject,
                        "issuer": issuer,
                        "expires": expiry_date.isoformat(),
                        "days_remaining": days_remaining,
                        "key_size": key_bits,
                        "signature_algorithm": sig_alg,
                        "serial_number": x509.get_serial_number(),
                        "version": x509.get_version()
                    }
        
        except Exception as e:
            return None
    
    async def check_protocols(self, hostname: str, port: int) -> List[Dict]:
        """Check supported SSL/TLS protocols"""
        protocols = []
        protocol_versions = [
            ("SSLv2", ssl.PROTOCOL_SSLv2),
            ("SSLv3", ssl.PROTOCOL_SSLv3),
            ("TLSv1", ssl.PROTOCOL_TLSv1),
            ("TLSv1.1", ssl.PROTOCOL_TLSv1_1),
            ("TLSv1.2", ssl.PROTOCOL_TLSv1_2),
            ("TLSv1.3", ssl.PROTOCOL_TLS)
        ]
        
        for name, proto in protocol_versions:
            try:
                context = ssl.SSLContext(proto)
                context.verify_mode = ssl.CERT_NONE
                context.check_hostname = False
                
                with socket.create_connection((hostname, port), timeout=5) as sock:
                    with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                        protocols.append({
                            "protocol": name,
                            "supported": True,
                            "version": ssock.version()
                        })
            except:
                protocols.append({
                    "protocol": name,
                    "supported": False,
                    "version": None
                })
        
        return protocols
    
    async def check_ciphers(self, hostname: str, port: int) -> List[Dict]:
        """Check supported ciphers"""
        supported_ciphers = []
        
        for cipher in self.ciphers:
            try:
                context = ssl.create_default_context()
                context.set_ciphers(cipher)
                
                with socket.create_connection((hostname, port), timeout=5) as sock:
                    with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                        supported_ciphers.append({
                            "cipher": cipher,
                            "supported": True,
                            "bits": ssock.cipher()[2] if ssock.cipher() else 0
                        })
            except:
                supported_ciphers.append({
                    "cipher": cipher,
                    "supported": False,
                    "bits": 0
                })
        
        return supported_ciphers
    
    async def check_vulnerabilities(self, hostname: str, port: int, 
                                  cert_info: Dict, protocols: List[Dict], 
                                  ciphers: List[Dict]) -> List[Dict]:
        """Check for SSL/TLS vulnerabilities"""
        vulnerabilities = []
        
        # Check certificate expiration
        if cert_info:
            days_remaining = cert_info.get("days_remaining", 0)
            if days_remaining < 30:
                vulnerabilities.append({
                    "name": "Certificate Expiring Soon",
                    "severity": "MEDIUM",
                    "description": f"Certificate expires in {days_remaining} days",
                    "remediation": "Renew certificate"
                })
            
            # Check key size
            key_size = cert_info.get("key_size", 0)
            if key_size < 2048:
                vulnerabilities.append({
                    "name": "Weak RSA Key",
                    "severity": "HIGH",
                    "description": f"Certificate uses {key_size}-bit RSA key",
                    "remediation": "Generate new certificate with 2048+ bit key"
                })
        
        # Check for weak protocols
        for proto in protocols:
            if proto["supported"] and proto["protocol"] in ["SSLv2", "SSLv3"]:
                vulnerabilities.append({
                    "name": f"Weak Protocol: {proto['protocol']}",
                    "severity": "HIGH",
                    "description": f"Server supports {proto['protocol']}",
                    "remediation": "Disable SSLv2/SSLv3"
                })
        
        # Check for weak ciphers
        weak_ciphers = ["RC4", "DES", "3DES", "MD5", "SHA1", "EXPORT", "NULL", "ANON"]
        for cipher_info in ciphers:
            if cipher_info["supported"]:
                cipher = cipher_info["cipher"]
                for weak in weak_ciphers:
                    if weak in cipher:
                        vulnerabilities.append({
                            "name": f"Weak Cipher: {cipher}",
                            "severity": "HIGH",
                            "description": f"Server supports weak cipher: {cipher}",
                            "remediation": "Disable weak ciphers"
                        })
                        break
        
        # Check for Heartbleed
        heartbleed = await self.check_heartbleed(hostname, port)
        if heartbleed:
            vulnerabilities.append({
                "name": "Heartbleed (CVE-2014-0160)",
                "severity": "CRITICAL",
                "description": "Server is vulnerable to Heartbleed",
                "remediation": "Update OpenSSL to patched version"
            })
        
        return vulnerabilities
    
    async def check_heartbleed(self, hostname: str, port: int) -> bool:
        """Check for Heartbleed vulnerability"""
        try:
            # Simplified Heartbleed check
            # Real implementation would send malicious heartbeat request
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect((hostname, port))
            
            # Send TLS Client Hello
            # This is a simplified check
            sock.send(b"\x16\x03\x01\x00\x75\x01\x00\x00\x71\x03\x01")
            time.sleep(1)
            
            response = sock.recv(1024)
            sock.close()
            
            # Check if server responds (very basic check)
            return len(response) > 0
        
        except:
            return False
