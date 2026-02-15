from ..deps import *

class PurePythonXSSScanner:
    """Pure Python XSS scanner"""
    
    def __init__(self):
        self.payloads = self.load_xss_payloads()
        self.contexts = ["html", "attribute", "javascript", "url"]
    
    def load_xss_payloads(self) -> Dict[str, List[str]]:
        """Load XSS payloads for different contexts"""
        return {
            "html": [
                "<script>alert(1)</script>",
                "<img src=x onerror=alert(1)>",
                "<svg onload=alert(1)>",
                "<body onload=alert(1)>",
                "<iframe src=javascript:alert(1)>"
            ],
            "attribute": [
                "\" onmouseover=\"alert(1)",
                "' onfocus='alert(1)",
                " onload=\"alert(1)\"",
                " autofocus onfocus=\"alert(1)\""
            ],
            "javascript": [
                "javascript:alert(1)",
                "jaVasCript:alert(1)",
                "jav&#x09;ascript:alert(1)",
                "javascript&#58;alert(1)"
            ],
            "url": [
                "http://evil.com",
                "//evil.com",
                "javascript:alert(document.domain)",
                "data:text/html,<script>alert(1)</script>"
            ]
        }
    
    async def scan(self, url: str, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Scan for XSS vulnerabilities"""
        
        results = {
            "url": url,
            "vulnerable": False,
            "reflected_xss": [],
            "stored_xss": [],  # Would need form submission to test
            "dom_xss": [],     # Would need JavaScript analysis
            "confidence": 0.0
        }
        
        if params is None:
            params = self.extract_parameters(url)
        
        # Test each parameter for reflected XSS
        for param_name, param_value in params.items():
            for context in self.contexts:
                xss_found = await self.test_xss(url, param_name, param_value, context)
                
                if xss_found["vulnerable"]:
                    results["vulnerable"] = True
                    results["reflected_xss"].append({
                        "parameter": param_name,
                        "context": context,
                        "payload": xss_found["payload"],
                        "confidence": xss_found["confidence"]
                    })
                    results["confidence"] = max(
                        results["confidence"], 
                        xss_found["confidence"]
                    )
        
        return results
    
    def extract_parameters(self, url: str) -> Dict[str, str]:
        """Extract parameters from URL"""
        params = {}
        
        try:
            parsed = urllib.parse.urlparse(url)
            query_params = urllib.parse.parse_qs(parsed.query)
            
            for key, values in query_params.items():
                if values:
                    params[key] = values[0]
        
        except Exception as e:
            pass
        
        return params
    
    async def test_xss(self, url: str, param: str, value: str, 
                      context: str) -> Dict[str, Any]:
        """Test for XSS in specific context"""
        
        result = {
            "vulnerable": False,
            "context": context,
            "parameter": param,
            "payload": "",
            "confidence": 0.0
        }
        
        # Get payloads for this context
        payloads = self.payloads.get(context, [])
        
        for payload in payloads:
            test_url = self.build_test_url(url, param, payload)
            response = await self.make_request(test_url)
            
            if not response:
                continue
            
            content = response.get("content", "")
            
            # Check if payload is reflected
            if payload in content:
                # Check if it's properly encoded
                encoded_payload = self.html_encode(payload)
                
                if encoded_payload not in content:
                    result["vulnerable"] = True
                    result["payload"] = payload
                    result["confidence"] = 0.8
                    
                    # Higher confidence if payload executes in certain contexts
                    if context == "html" and "<script>" in payload:
                        result["confidence"] = 0.9
                    
                    break
        
        return result
    
    def build_test_url(self, url: str, param: str, payload: str) -> str:
        """Build test URL with payload"""
        parsed = urllib.parse.urlparse(url)
        query_params = urllib.parse.parse_qs(parsed.query)
        
        # Update parameter with payload
        query_params[param] = [payload]
        
        # Rebuild URL
        new_query = urllib.parse.urlencode(query_params, doseq=True)
        new_url = urllib.parse.urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment
        ))
        
        return new_url
    
    async def make_request(self, url: str) -> Optional[Dict]:
        """Make HTTP request"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    content = await response.text()
                    
                    return {
                        "url": url,
                        "status": response.status,
                        "content": content,
                        "headers": dict(response.headers)
                    }
        
        except Exception as e:
            return None
    
    def html_encode(self, text: str) -> str:
        """HTML encode text"""
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#39;'))
